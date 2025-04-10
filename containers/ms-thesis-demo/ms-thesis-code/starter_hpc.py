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

# ====================== KONFİGÜRASYON ======================
NUM_PLAYERS = 6
PLAYER_FEATURES = 4  # x, y, z, team
HIDDEN_DIM = 32

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

# ====================== VERİ YÜKLEME ======================
def load_and_process_data(csv_path):
    df = pd.read_csv(csv_path)
    dataset = []

    for idx, row in df.iterrows():
        if row.isnull().any():
            continue

        x = []
        for i in range(NUM_PLAYERS):
            pos = [
                float(row[f'p{i}_pos_x']),
                float(row[f'p{i}_pos_y']),
                float(row[f'p{i}_pos_z']),
                float(row[f'p{i}_team']),
            ]
            x.append(pos)

        x = torch.tensor(x, dtype=torch.float32)

        edge_index = []
        edge_weights = []
        for i in range(NUM_PLAYERS):
            for j in range(NUM_PLAYERS):
                if i != j:
                    dist = torch.norm(x[i, :3] - x[j, :3])
                    weight = 1.0 / (1.0 + dist)
                    if torch.isnan(weight):
                        weight = 0.0
                    edge_weights.append(weight)
                    edge_index.append([i, j])

        data = Data(
            x=x,
            edge_index=torch.tensor(edge_index).t().contiguous(),
            edge_weight=torch.tensor(edge_weights, dtype=torch.float32),
            orange_y=torch.tensor([float(row['team_0_goal_prev_5s'])], dtype=torch.float32),
            blue_y=torch.tensor([float(row['team_1_goal_prev_5s'])], dtype=torch.float32)
        )
        dataset.append(data)

    return dataset

# ====================== MODEL ======================
class SafeRocketLeagueGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(PLAYER_FEATURES, HIDDEN_DIM)
        self.conv2 = GCNConv(HIDDEN_DIM, HIDDEN_DIM)

        self.orange_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 1),
            nn.Sigmoid()
        )
        self.blue_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = global_mean_pool(x, data.batch)

        orange_out = self.orange_head(x)
        blue_out = self.blue_head(x)

        if torch.isnan(orange_out).any() or torch.isnan(blue_out).any():
            print("NaN detected in outputs!")
            print("x:", x)
            raise ValueError("NaN in model output")

        return orange_out, blue_out

# ====================== EĞİTİM ======================
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
            optimizer.zero_grad()
            orange_pred, blue_pred = model(batch)

            loss_orange = criterion(orange_pred, batch.orange_y.unsqueeze(1))
            loss_blue = criterion(blue_pred, batch.blue_y.unsqueeze(1))
            loss = loss_orange + loss_blue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            correct_orange += ((orange_pred > 0.5).float() == batch.orange_y.unsqueeze(1)).sum().item()
            correct_blue += ((blue_pred > 0.5).float() == batch.blue_y.unsqueeze(1)).sum().item()

        model.eval()
        test_orange = test_blue = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                orange_pred, blue_pred = model(batch)
                test_orange += ((orange_pred > 0.5).float() == batch.orange_y.unsqueeze(1)).sum().item()
                test_blue += ((blue_pred > 0.5).float() == batch.blue_y.unsqueeze(1)).sum().item()

        wandb.log({
            'epoch': epoch,
            'train_loss': total_loss / len(train_loader),
            'train_acc_orange': correct_orange / len(train_data),
            'train_acc_blue': correct_blue / len(train_data),
            'test_acc_orange': test_orange / len(test_data),
            'test_acc_blue': test_blue / len(test_data),
        })

        print(f"Epoch {epoch:03d} | Loss: {total_loss/len(train_loader):.4f} | "
              f"Train Acc: O {correct_orange/len(train_data):.3f}, B {correct_blue/len(train_data):.3f} | "
              f"Test Acc: O {test_orange/len(test_data):.3f}, B {test_blue/len(test_data):.3f}")

    wandb.finish()

if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    torch.autograd.set_detect_anomaly(True)
    main()
