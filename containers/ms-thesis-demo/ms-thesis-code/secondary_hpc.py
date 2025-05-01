import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import pandas as pd
from sklearn.model_selection import train_test_split
import wandb
import os
import argparse

# ====================== CONFIGURATION ======================
NUM_PLAYERS = 6
PLAYER_FEATURES = 13  # xyz_pos, xyz_vel, xyz_forward, boost, team, alive, distance_to_ball
HIDDEN_DIM = 32
GLOBAL_FEATURE_DIM = 9  # xyz_ball_pos, xyz_ball_vel, boost_pad_respawn_times, ball_hit_team_num, seconds_remaining


# Normalization bounds
POS_MIN_X, POS_MAX_X = -4096, 4096  # X bounds for a standard map 
POS_MIN_Y, POS_MAX_Y = -6000, 6000  # Y bounds for a standard map
POS_MIN_Z, POS_MAX_Z = 0, 2044      # Z bounds for a standard map
VEL_MIN, VEL_MAX = -2300, 2300  # Car velocity bounds for a standard map
BOOST_MIN, BOOST_MAX = 0, 100  # Boost amount bounds for a standard map
BALL_VEL_MIN, BALL_VEL_MAX = -6000, 6000  # Ball velocity bounds for a standard map
BOOST_PAD_MIN, BOOST_PAD_MAX = 0, 10  # Boost pad respawn time bounds for a standard map
DIST_MIN, DIST_MAX = 0, (8192**2 + 10240**2 + 2044**2)**(1/2) # Max possible distance between two entities

def parse_args():
    parser = argparse.ArgumentParser(description="Rocket League GCN")
    parser.add_argument('--csv-path', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()

# ====================== DATA LOADING ======================
def load_and_process_data(csv_path):
    df = pd.read_csv(csv_path)
    dataset = []

    for idx, row in df.iterrows():
        if row.isnull().any():
            continue  # Skip rows with NaN
        
        # === Normalize Player Node Features ===
        x = []
        for i in range(NUM_PLAYERS):
            # Normalize positions to [0, 1]
            pos_x = (float(row[f'p{i}_pos_x']) - POS_MIN_X) / (POS_MAX_X - POS_MIN_X)
            pos_y = (float(row[f'p{i}_pos_y']) - POS_MIN_Y) / (POS_MAX_Y - POS_MIN_Y)
            pos_z = (float(row[f'p{i}_pos_z']) - POS_MIN_Z) / (POS_MAX_Z - POS_MIN_Z)
            # Team (already binary, no scaling needed)
            team = float(row[f'p{i}_team'])
            x.append([pos_x, pos_y, pos_z, team])  # Now all features in [0, 1]

        x = torch.tensor(x, dtype=torch.float32)

        # === Edge Construction (Unchanged) ===
        edge_index = []
        edge_weights = []
        for i in range(NUM_PLAYERS):
            for j in range(NUM_PLAYERS):
                if i != j:
                    dist = torch.norm(x[i, :3] - x[j, :3])  # Distance between normalized positions
                    weight = 1.0 / (1.0 + dist)
                    if torch.isnan(weight):
                        weight = 0.0
                    edge_weights.append(weight)
                    edge_index.append([i, j])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float32)

        # === Normalize Global Features to [0, 1] ===
        global_features = torch.tensor([
            (float(row['ball_dist_to_goal1']) - DIST_MIN) / (DIST_MAX - DIST_MIN),
            (float(row['ball_dist_to_goal2']) - DIST_MIN) / (DIST_MAX - DIST_MIN),
        ], dtype=torch.float32)

        # === Labels (Unchanged) ===
        orange_y = torch.tensor([float(row['team_0_goal_prev_5s'])], dtype=torch.float32)
        blue_y = torch.tensor([float(row['team_1_goal_prev_5s'])], dtype=torch.float32)

        # === Data Object (Now with Normalized Features) ===
        data = Data(
            x=x,  # Normalized to [0, 1]
            edge_index=edge_index,
            edge_weight=edge_weights,
            global_features=global_features.unsqueeze(0),  # Normalized to [0, 1]
            orange_y=orange_y,
            blue_y=blue_y
        )
        dataset.append(data)

    return dataset

# ====================== MODEL ======================
class SafeRocketLeagueGCN(nn.Module):
    def __init__(self):
        super().__init__()
        # Layer 1: Conv → BN → ReLU
        self.conv1 = GCNConv(PLAYER_FEATURES, HIDDEN_DIM)
        self.bn1 = nn.BatchNorm1d(HIDDEN_DIM)  # <-- BatchNorm after conv1
        # Layer 2: Conv → BN → ReLU
        self.conv2 = GCNConv(HIDDEN_DIM, HIDDEN_DIM)
        self.bn2 = nn.BatchNorm1d(HIDDEN_DIM)  # <-- BatchNorm after conv2
        # Heads (unchanged)
        self.orange_head = nn.Sequential(nn.Linear(...), nn.Sigmoid())
        self.blue_head = nn.Sequential(nn.Linear(...), nn.Sigmoid())

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        
        # Layer 1: Conv → BN → ReLU
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)  # <-- Normalize activations
        x = F.relu(x)
        
        # Layer 2: Conv → BN → ReLU
        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn2(x)  # <-- Normalize activations
        x = F.relu(x)
        
        # Rest of the model (unchanged)
        x = global_mean_pool(x, data.batch)
        x = torch.cat([x, data.global_features], dim=1)
        return self.orange_head(x), self.blue_head(x)
    
# ====================== LOGGING ======================
def log_feature_importance_to_wandb(node_feature_grads, global_feature_grads, epoch):
    # You might want to log both the per-player (node) features and the global features
    wandb.log({
        'epoch': epoch,
        'node_feature_importance': node_feature_grads.cpu().numpy(),
        'global_feature_importance': global_feature_grads.cpu().numpy(),
    })

# ====================== TRAINING ======================
def main():
    args = parse_args()

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(project="rocket-league-gcn-safe", config=args)

    dataset = load_and_process_data(args.csv_path)
    train_data, test_data = train_test_split(dataset, test_size=args.test_size, random_state=args.random_seed)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SafeRocketLeagueGCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCELoss()

    max_grad_norm = 1.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct_orange = 0
        correct_blue = 0

        for batch in train_loader:
            batch = batch.to(device)
            
            # Enable gradient tracking
            batch.x.requires_grad_(True)
            batch.global_features.requires_grad_(True)
            
            optimizer.zero_grad()
            orange_pred, blue_pred = model(batch)
            
            loss_orange = criterion(orange_pred, batch.orange_y.unsqueeze(1))
            loss_blue = criterion(blue_pred, batch.blue_y.unsqueeze(1))
            loss = loss_orange + loss_blue
            
            loss.backward()
            optimizer.step()

            # Calculate feature importance
            node_feature_grads = batch.x.grad.abs().mean(dim=0)
            global_feature_grads = batch.global_features.grad.abs().mean(dim=0)

            # Log to WandB
            log_feature_importance_to_wandb(node_feature_grads, global_feature_grads, epoch)

            total_loss += loss.item()
            correct_orange += ((orange_pred > 0.5).float() == batch.orange_y.unsqueeze(1)).sum().item()
            correct_blue += ((blue_pred > 0.5).float() == batch.blue_y.unsqueeze(1)).sum().item()

        # WandB metrics for loss and accuracy
        wandb.log({
            'epoch': epoch,
            'train_loss': total_loss / len(train_loader),
            'train_acc_orange': correct_orange / len(train_data),
            'train_acc_blue': correct_blue / len(train_data),
        })

        print(f"Epoch {epoch:03d} | Loss: {total_loss/len(train_loader):.4f} | "
            f"Train Acc: O {correct_orange/len(train_data):.3f}, B {correct_blue/len(train_data):.3f}")

    wandb.finish()



if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    torch.autograd.set_detect_anomaly(True)
    main()
