import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import wandb
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

# ====================== CONFIGURATION ======================
NUM_PLAYERS = 6
PLAYER_FEATURES = 13  # xyz_pos, xyz_vel, xyz_forward, boost, team, alive, distance_to_ball
HIDDEN_DIM = 32
GLOBAL_FEATURE_DIM = 9  # xyz_ball_pos, xyz_ball_vel, boost_pad_respawn_times, ball_hit_team_num, seconds_remaining

# Normalization bounds
POS_MIN_X, POS_MAX_X = -4096, 4096
POS_MIN_Y, POS_MAX_Y = -6000, 6000
POS_MIN_Z, POS_MAX_Z = 0, 2044
VEL_MIN, VEL_MAX = -2300, 2300
BOOST_MIN, BOOST_MAX = 0, 100
BALL_VEL_MIN, BALL_VEL_MAX = -6000, 6000
BOOST_PAD_MIN, BOOST_PAD_MAX = 0, 10
DIST_MIN, DIST_MAX = 0, (8192**2 + 10240**2 + 2044**2)**(1/2)


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
def normalize(val, min_val, max_val):
    return (val - min_val) / (max_val - min_val + 1e-8)

def load_and_process_data(csv_path):
    df = pd.read_csv(csv_path)
    dataset = []

    for idx, row in df.iterrows():
        if row.isnull().any():
            continue

        x = []
        for i in range(NUM_PLAYERS):
            pos_x = normalize(float(row[f'p{i}_pos_x']), POS_MIN_X, POS_MAX_X)
            pos_y = normalize(float(row[f'p{i}_pos_y']), POS_MIN_Y, POS_MAX_Y)
            pos_z = normalize(float(row[f'p{i}_pos_z']), POS_MIN_Z, POS_MAX_Z)
            vel_x = normalize(float(row[f'p{i}_vel_x']), VEL_MIN, VEL_MAX)
            vel_y = normalize(float(row[f'p{i}_vel_y']), VEL_MIN, VEL_MAX)
            vel_z = normalize(float(row[f'p{i}_vel_z']), VEL_MIN, VEL_MAX)
            forward_x = float(row[f'p{i}_forward_x'])
            forward_y = float(row[f'p{i}_forward_y'])
            forward_z = float(row[f'p{i}_forward_z'])
            boost = normalize(float(row[f'p{i}_boost_amount']), BOOST_MIN, BOOST_MAX)
            team = float(row[f'p{i}_team'])
            alive = float(row[f'p{i}_alive'])
            dist_to_ball = normalize(float(row[f'p{i}_dist_to_ball']), DIST_MIN, DIST_MAX)

            x.append([
                pos_x, pos_y, pos_z,
                vel_x, vel_y, vel_z,
                forward_x, forward_y, forward_z,
                boost, team, alive, dist_to_ball
            ])

        x = torch.tensor(x, dtype=torch.float32)

        edge_index = []
        edge_weights = []
        for i in range(NUM_PLAYERS):
            for j in range(NUM_PLAYERS):
                if i != j:
                    dist = torch.norm(x[i, :3] - x[j, :3])
                    weight = 1.0 / (1.0 + dist)
                    edge_weights.append(weight if not torch.isnan(weight) else 0.0)
                    edge_index.append([i, j])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float32)

        seconds_remaining = float(row['seconds_remaining'])
        normalized_seconds = normalize(min(seconds_remaining, 300.0), 0, 300)

        global_features = torch.tensor([
            normalize(float(row['ball_pos_x']), POS_MIN_X, POS_MAX_X),
            normalize(float(row['ball_pos_y']), POS_MIN_Y, POS_MAX_Y),
            normalize(float(row['ball_pos_z']), POS_MIN_Z, POS_MAX_Z),
            normalize(float(row['ball_vel_x']), BALL_VEL_MIN, BALL_VEL_MAX),
            normalize(float(row['ball_vel_y']), BALL_VEL_MIN, BALL_VEL_MAX),
            normalize(float(row['ball_vel_z']), BALL_VEL_MIN, BALL_VEL_MAX),
            normalize(float(row['boost_pad_0_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX),
            float(row['ball_hit_team_num']),
            normalized_seconds
        ], dtype=torch.float32)

        orange_y = torch.tensor([float(row['team_0_goal_prev_5s'])], dtype=torch.float32)
        blue_y = torch.tensor([float(row['team_1_goal_prev_5s'])], dtype=torch.float32)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weights,
            global_features=global_features.unsqueeze(0),
            orange_y=orange_y,
            blue_y=blue_y
        )
        dataset.append(data)

    return dataset


# ====================== MODEL ======================
class SafeRocketLeagueGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(PLAYER_FEATURES, HIDDEN_DIM)
        self.bn1 = nn.BatchNorm1d(HIDDEN_DIM)
        self.conv2 = GCNConv(HIDDEN_DIM, HIDDEN_DIM)
        self.bn2 = nn.BatchNorm1d(HIDDEN_DIM)
        self.orange_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM + GLOBAL_FEATURE_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.blue_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM + GLOBAL_FEATURE_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        x = global_mean_pool(x, data.batch)
        x = torch.cat([x, data.global_features], dim=1)
        return self.orange_head(x), self.blue_head(x)


# ====================== LOGGING ======================
def log_feature_importance_to_wandb(node_feature_grads, global_feature_grads, epoch):
    wandb.log({
        'epoch': epoch,
        'node_feature_importance': wandb.Histogram(node_feature_grads.cpu().numpy()),
        'global_feature_importance': wandb.Histogram(global_feature_grads.cpu().numpy()),
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

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct_orange = 0
        correct_blue = 0
        all_orange_preds = []
        all_orange_labels = []
        all_blue_preds = []
        all_blue_labels = []

        for batch in train_loader:
            batch = batch.to(device)
            batch.x.requires_grad_(True)
            batch.global_features.requires_grad_(True)
            optimizer.zero_grad()
            orange_pred, blue_pred = model(batch)
            loss_orange = criterion(orange_pred, batch.orange_y.unsqueeze(1))
            loss_blue = criterion(blue_pred, batch.blue_y.unsqueeze(1))
            loss = loss_orange + loss_blue
            loss.backward()
            optimizer.step()

            node_feature_grads = batch.x.grad.abs().mean(dim=0)
            global_feature_grads = batch.global_features.grad.abs().mean(dim=0)
            log_feature_importance_to_wandb(node_feature_grads, global_feature_grads, epoch)

            total_loss += loss.item()
            pred_orange = (orange_pred > 0.5).float()
            pred_blue = (blue_pred > 0.5).float()

            correct_orange += (pred_orange == batch.orange_y.unsqueeze(1)).sum().item()
            correct_blue += (pred_blue == batch.blue_y.unsqueeze(1)).sum().item()

            all_orange_preds.extend(pred_orange.cpu().numpy())
            all_orange_labels.extend(batch.orange_y.cpu().numpy())
            all_blue_preds.extend(pred_blue.cpu().numpy())
            all_blue_labels.extend(batch.blue_y.cpu().numpy())

        orange_f1 = f1_score(all_orange_labels, all_orange_preds)
        blue_f1 = f1_score(all_blue_labels, all_blue_preds)
        orange_precision = precision_score(all_orange_labels, all_orange_preds)
        blue_precision = precision_score(all_blue_labels, all_blue_preds)
        orange_recall = recall_score(all_orange_labels, all_orange_preds)
        blue_recall = recall_score(all_blue_labels, all_blue_preds)
        orange_cm = confusion_matrix(all_orange_labels, all_orange_preds).tolist()
        blue_cm = confusion_matrix(all_blue_labels, all_blue_preds).tolist()

        wandb.log({
            'epoch': epoch,
            'train_loss': total_loss / len(train_loader),
            'train_acc_orange': correct_orange / len(train_data),
            'train_acc_blue': correct_blue / len(train_data),
            'orange_f1': orange_f1,
            'blue_f1': blue_f1,
            'orange_precision': orange_precision,
            'blue_precision': blue_precision,
            'orange_recall': orange_recall,
            'blue_recall': blue_recall,
            'orange_confusion_matrix': wandb.plot.confusion_matrix(y_true=all_orange_labels, preds=all_orange_preds, class_names=["No Goal", "Goal"]),
            'blue_confusion_matrix': wandb.plot.confusion_matrix(y_true=all_blue_labels, preds=all_blue_preds, class_names=["No Goal", "Goal"])
        })

        print(f"Epoch {epoch:03d} | Loss: {total_loss/len(train_loader):.4f} | "
              f"Train Acc: O {correct_orange/len(train_data):.3f}, B {correct_blue/len(train_data):.3f} | "
              f"F1: O {orange_f1:.3f}, B {blue_f1:.3f} | "
              f"Precision: O {orange_precision:.3f}, B {blue_precision:.3f} | "
              f"Recall: O {orange_recall:.3f}, B {blue_recall:.3f}")

    wandb.finish()


if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    torch.autograd.set_detect_anomaly(True)
    main()
