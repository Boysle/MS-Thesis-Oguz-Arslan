import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import networkx as nx
from sklearn.manifold import TSNE
import os
import wandb
import argparse

# ====================== CONFIGURATION ======================
NUM_PLAYERS = 6
PLAYER_FEATURES = 3  # x,y,z coordinates
HIDDEN_DIM = 32

def parse_args():
    parser = argparse.ArgumentParser(description="Rocket League GCN")
    parser.add_argument('--csv-path', type=str, required=True, 
                       help='Path to the CSV file containing replay data')
    parser.add_argument('--batch-size', type=int, default=32, 
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30, 
                       help='Number of training epochs')
    parser.add_argument('--test-size', type=float, default=0.2, 
                       help='Test set size (0 to 1)')
    parser.add_argument('--random-seed', type=int, default=42, 
                       help='Random seed for reproducibility')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate for optimizer')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with verbose output')
    parser.add_argument('--visualize', action='store_true',
                       help='Enable visualization during training')
    return parser.parse_args()

# ====================== DATA PROCESSING ======================
def load_and_process_data(csv_path):
    """Load single CSV file and convert to PyG Data objects"""
    df = pd.read_csv(csv_path)
    
    dataset = []
    for idx, row in df.iterrows():
        # Extract node features
        x = []
        team_labels = []
        for i in range(NUM_PLAYERS):
            pos = [row[f'p{i}_pos_x'], row[f'p{i}_pos_y'], row[f'p{i}_pos_z']]
            team = row[f'p{i}_team']
            x.append(pos)
            team_labels.append(team)
        
        x = torch.tensor(x, dtype=torch.float)
        team_labels = torch.tensor(team_labels, dtype=torch.long)
        
        # Calculate edge weights
        edge_index = []
        edge_weights = []
        for i in range(NUM_PLAYERS):
            for j in range(NUM_PLAYERS):
                if i != j:
                    distance = torch.norm(x[i] - x[j])
                    edge_index.append([i, j])
                    edge_weights.append(1.0 / (1.0 + distance))
        
        # Create graph data object
        data = Data(
            x=x,
            edge_index=torch.tensor(edge_index).t().contiguous(),
            edge_weight=torch.tensor(edge_weights),
            orange_y=torch.tensor([row['team_0_goal_prev_5s']], dtype=torch.long),
            blue_y=torch.tensor([row['team_1_goal_prev_5s']], dtype=torch.long),
            team_labels=team_labels,
            game_id=idx
        )
        dataset.append(data)
    
    return dataset

# ====================== MODEL ARCHITECTURE ======================
class RocketLeagueGCN(nn.Module):
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
        x = F.relu(self.conv1(data.x, data.edge_index, data.edge_weight))
        x = F.relu(self.conv2(x, data.edge_index, data.edge_weight))
        x = global_mean_pool(x, data.batch)
        return self.orange_head(x), self.blue_head(x)

# ====================== MAIN EXECUTION ======================
if __name__ == "__main__":
    args = parse_args()
    
    # Initialize W&B
    wandb.login(key=os.getenv('WANDB_API_KEY'))
    wandb.init(project="rocket-league-gcn", config=args)
    config = wandb.config

    # Load and process data
    dataset = load_and_process_data(config.csv_path)
    train_data, test_data = train_test_split(
        dataset, test_size=config.test_size, random_state=config.random_seed
    )
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config.batch_size)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RocketLeagueGCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCELoss()

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        total_loss = correct_orange = correct_blue = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            orange_pred, blue_pred = model(batch)
            loss = criterion(orange_pred, batch.orange_y.float().unsqueeze(1)) + \
                   criterion(blue_pred, batch.blue_y.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct_orange += ((orange_pred > 0.5) == batch.orange_y.unsqueeze(1)).sum().item()
            correct_blue += ((blue_pred > 0.5) == batch.blue_y.unsqueeze(1)).sum().item()

        # Log metrics
        metrics = {
            'epoch': epoch,
            'train_loss': total_loss/len(train_loader),
            'train_acc_orange': correct_orange/len(train_data),
            'train_acc_blue': correct_blue/len(train_data),
        }
        
        # Validation
        model.eval()
        with torch.no_grad():
            test_orange = test_blue = 0
            for batch in test_loader:
                batch = batch.to(device)
                orange_pred, blue_pred = model(batch)
                test_orange += ((orange_pred > 0.5) == batch.orange_y.unsqueeze(1)).sum().item()
                test_blue += ((blue_pred > 0.5) == batch.blue_y.unsqueeze(1)).sum().item()
            
            metrics.update({
                'test_acc_orange': test_orange/len(test_data),
                'test_acc_blue': test_blue/len(test_data)
            })
        
        wandb.log(metrics)
        print(f"Epoch {epoch:03d} | Loss: {metrics['train_loss']:.4f} | "
              f"Train: O {metrics['train_acc_orange']:.3f} B {metrics['train_acc_blue']:.3f} | "
              f"Test: O {metrics['test_acc_orange']:.3f} B {metrics['test_acc_blue']:.3f}")

    wandb.finish()
