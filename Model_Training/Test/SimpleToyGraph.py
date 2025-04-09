import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch import nn, optim
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_networkx
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import wandb

# Initialize W&B
wandb.init(project="rl-gcn-stanford-style")

# --------------------------
# 1. Dataset Class (Fixed Imports)
# --------------------------
class RocketLeagueDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.csv_files = [f for f in os.listdir(self.raw_dir) if f.endswith('.csv')]
        self.process()

    @property
    def raw_file_names(self):
        return self.csv_files

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(len(self.csv_files))]

    def process(self):
        for i, csv_file in enumerate(self.csv_files):
            df = pd.read_csv(os.path.join(self.raw_dir, csv_file))
            data = self.csv_to_graph(df.iloc[0])  # Using first frame only
            torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))

    def csv_to_graph(self, row):
        # Node features [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, boost, team]
        node_features = torch.zeros((6, 8), dtype=torch.float32)
        for i in range(6):
            node_features[i] = torch.tensor([
                row[f'p{i}_pos_x'], row[f'p{i}_pos_y'], row[f'p{i}_pos_z'],
                row[f'p{i}_vel_x'], row[f'p{i}_vel_y'], row[f'p{i}_vel_z'],
                row[f'p{i}_boost_amount'],
                row[f'p{i}_team']
            ])

        # Edges (fully connected)
        edge_index = []
        for i in range(6):
            for j in range(6):
                if i != j:
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # State and labels
        state = torch.tensor([
            row['ball_pos_x'], row['ball_pos_y'], row['ball_pos_z'],
            row['ball_vel_x'], row['ball_vel_y'], row['ball_vel_z'],
            row['seconds_remaining'], row['is_overtime']
        ], dtype=torch.float32)

        labels = torch.tensor([
            row['team_0_goal_prev_5s'],
            row['team_1_goal_prev_5s']
        ], dtype=torch.float32)

        return Data(x=node_features, edge_index=edge_index, state=state, y=labels)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))

# --------------------------
# 2. GCN Model
# --------------------------
class RLGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(8, 64)
        self.conv2 = GCNConv(64, 64)
        self.state_mlp = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU()
        )
        self.classifier = nn.Linear(128, 2)

    def forward(self, data):
        # Graph processing
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, data.edge_index)
        x = global_mean_pool(x, batch=torch.zeros(data.num_nodes, dtype=torch.long))
        
        # State processing
        state = self.state_mlp(data.state)
        
        # Combined prediction
        return self.classifier(torch.cat([x, state], dim=1))

# --------------------------
# 3. Visualization
# --------------------------
def visualize_graph(data):
    G = to_networkx(data, to_undirected=True)
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(G)
    
    # Node colors by team
    node_colors = ['blue' if data.x[i, -1] == 0 else 'orange' for i in range(6)]
    nx.draw(G, pos, node_color=node_colors, with_labels=False, node_size=300)
    
    # Add ball
    plt.scatter([0], [0], c='red', s=500, marker='*')
    plt.title(f"Frame Visualization\nBall Vel: {data.state[3:6].tolist()}")
    plt.show()

# --------------------------
# 4. Training
# --------------------------
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset
    dataset = RocketLeagueDataset(root='E:\\Raw RL Esports Replays\\Day 3 Swiss Stage')
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Model
    model = RLGCN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # Training loop
    for epoch in range(1, 101):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = F.binary_cross_entropy_with_logits(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Evaluation
        model.eval()
        correct = 0
        for batch in test_loader:
            batch = batch.to(device)
            pred = (torch.sigmoid(model(batch)) > 0.5).float()
            correct += (pred == batch.y).all(dim=1).sum().item()
        
        acc = correct / len(test_loader.dataset)
        print(f'Epoch: {epoch:03d}, Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.4f}')
        wandb.log({"epoch": epoch, "loss": total_loss/len(train_loader), "accuracy": acc})
    
    # Visualize sample
    sample = dataset[0]
    visualize_graph(sample)
    torch.save(model.state_dict(), "rl_gcn_model.pth")

if __name__ == "__main__":
    train()