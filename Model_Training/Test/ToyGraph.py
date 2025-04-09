import os
import torch
import pandas as pd
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch import nn, optim
import numpy as np
from torch_geometric.nn import GCNConv, global_mean_pool
from tqdm import tqdm
import wandb

wandb.init(project="rl-goal-prediction", config={
    "seq_length": 10,
    "stride": 5,
    "batch_size": 32,
    "learning_rate": 0.001
})

path = "E:\\Raw RL Esports Replays\\Day 3 Swiss Stage\\"
if not os.path.exists(path):
    raise ValueError(f"Path not found: {path}")

# --------------------------
# 1. Data Loading Utilities
# --------------------------
def find_all_csv_files(data_root):
    # This creates a list of all CSV file paths
    return [os.path.join(root, f)          # Combine folder + filename to make full path
            for root, _, files in os.walk(data_root)  # Look through all folders
            for f in files if f.endswith('.csv')]  # Only keep files ending with .csv

def load_csv_to_dataframe(csv_path):
    """Load CSV with optimized data types to reduce memory."""
    dtype_map = {
    # --- Integers ---
    'is_overtime': 'int8',
    'p0_team': 'int8', 'p1_team': 'int8', 'p2_team': 'int8',
    'p3_team': 'int8', 'p4_team': 'int8', 'p5_team': 'int8',
    'team_0_goal_prev_5s': 'int8', 'team_1_goal_prev_5s': 'int8',
    
    # --- Half Precision Floats ---
    'p0_boost_amount': 'float16', 'p1_boost_amount': 'float16',
    'p2_boost_amount': 'float16', 'p3_boost_amount': 'float16',
    'p4_boost_amount': 'float16', 'p5_boost_amount': 'float16',
    
    # --- Full Precision Floats (physics-critical) ---
    # Player Positions
    'p0_pos_x': 'float32', 'p0_pos_y': 'float32', 'p0_pos_z': 'float32',
    'p1_pos_x': 'float32', 'p1_pos_y': 'float32', 'p1_pos_z': 'float32',
    'p2_pos_x': 'float32', 'p2_pos_y': 'float32', 'p2_pos_z': 'float32',
    'p3_pos_x': 'float32', 'p3_pos_y': 'float32', 'p3_pos_z': 'float32',
    'p4_pos_x': 'float32', 'p4_pos_y': 'float32', 'p4_pos_z': 'float32',
    'p5_pos_x': 'float32', 'p5_pos_y': 'float32', 'p5_pos_z': 'float32',
    
    # Player Velocities
    'p0_vel_x': 'float32', 'p0_vel_y': 'float32', 'p0_vel_z': 'float32',
    'p1_vel_x': 'float32', 'p1_vel_y': 'float32', 'p1_vel_z': 'float32',
    'p2_vel_x': 'float32', 'p2_vel_y': 'float32', 'p2_vel_z': 'float32',
    'p3_vel_x': 'float32', 'p3_vel_y': 'float32', 'p3_vel_z': 'float32',
    'p4_vel_x': 'float32', 'p4_vel_y': 'float32', 'p4_vel_z': 'float32',
    'p5_vel_x': 'float32', 'p5_vel_y': 'float32', 'p5_vel_z': 'float32',
    
    # Ball Data
    'ball_pos_x': 'float32', 'ball_pos_y': 'float32', 'ball_pos_z': 'float32',
    'ball_vel_x': 'float32', 'ball_vel_y': 'float32', 'ball_vel_z': 'float32',
    
    # Time (keep full precision for sub-second accuracy)
    'time': 'float32',
    'seconds_remaining': 'float32'
    }
    return pd.read_csv(csv_path, dtype=dtype_map)

# --------------------------
# 2. Graph Data Construction
# --------------------------
def dataframe_to_graph(df_frame):
    """Convert DataFrame row to graph Data object."""
    row = df_frame.iloc[0] if isinstance(df_frame, pd.DataFrame) else df_frame
    
    # Node features [pos, vel, boost, team]
    node_features = torch.zeros((6, 8), dtype=torch.float32)  # 6 players × 8 features each
    for i in range(6):
        node_features[i] = torch.tensor([
            row[f'p{i}_pos_x'], row[f'p{i}_pos_y'], row[f'p{i}_pos_z'],  # Position (3D)
            row[f'p{i}_vel_x'], row[f'p{i}_vel_y'], row[f'p{i}_vel_z'],  # Velocity (3D)
            row[f'p{i}_boost_amount'],  # Boost (float16)
            row[f'p{i}_team']           # Team ID (0 or 1)
        ])

    # Edge indices and distances
    edge_index = []  # Stores player connections
    edge_attr = []   # Stores distances

    for i in range(6):
        for j in range(6):
            if i != j:  # No self-connections
                dist = torch.norm(node_features[i, :3] - node_features[j, :3])  # 3D distance
                edge_index.append([i, j])
                edge_attr.append(dist)
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32).unsqueeze(1)

    # State vector and labels
    state_vector = torch.tensor([
        row['ball_pos_x'], row['ball_pos_y'], row['ball_pos_z'],  # Ball position
        row['ball_vel_x'], row['ball_vel_y'], row['ball_vel_z'],  # Ball velocity
        row['seconds_remaining'], row['is_overtime']              # Time info
    ], dtype=torch.float32)

    labels = torch.tensor([
        row['team_0_goal_prev_5s'],  # Will blue team score within 5 seconds?
        row['team_1_goal_prev_5s']   # Will orange team score within 5 seconds?
    ], dtype=torch.float32)

    return Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        state=state_vector,
        y=labels
    )

# --------------------------
# 3. Dataset Class
# --------------------------
class SequenceDataset(Dataset):
    def __init__(self, data_root, seq_length=10, stride=5):
        super().__init__()  # Crucial PyG initialization
        self.csv_files = find_all_csv_files(data_root)
        self.seq_length = seq_length
        self.stride = stride
        
        # Pre-compute all valid sequence starts
        self.sequence_starts = []
        for csv_path in self.csv_files:
            with open(csv_path) as f:
                num_frames = sum(1 for _ in f) - 1  # Subtract header
            
            # Add all valid starting positions for this CSV
            self.sequence_starts.extend([
                (csv_path, start) 
                for start in range(0, num_frames - seq_length + 1, stride)
            ])

    def len(self):
        return len(self.sequence_starts)  # Total number of sequences

    def get(self, idx):
        csv_path, start = self.sequence_starts[idx]
        df = load_csv_to_dataframe(csv_path)
        return [dataframe_to_graph(df.iloc[i]) 
                for i in range(start, start + self.seq_length)]

# --------------------------
# 4. GCN + State Joint Model
# --------------------------
class GCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(8, 64)
        self.conv2 = GCNConv(64, 64)
        self.state_mlp = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU()
        )
        self.output = nn.Linear(64 + 64, 2)  # 64 graph + 64 state

    def forward_single(self, graph):
        x = F.relu(self.conv1(graph.x, graph.edge_index))
        x = self.conv2(x, graph.edge_index)
        graph_embed = global_mean_pool(x, batch=None)
        state_embed = self.state_mlp(graph.state)
        return torch.cat([graph_embed, state_embed], dim=1)
    
    def forward(self, sequences):
        # sequences: List of graph sequences
        return torch.stack([self.output(self.forward_single(g)) 
                          for g in sequences[-1]])  # Only predict last frame

# --------------------------
# 5. Training Loop
# --------------------------
def train():
    # 1. Initialize Dataset
    full_dataset = SequenceDataset(path, seq_length=10)

    # Check if CSV files are actually being found
    print("First 5 CSV paths:")
    for f in full_dataset.csv_files[:5]:
        print(f"  {f}")
        if not os.path.exists(f):
            print("    ^^ WARNING: File not found!")
    
    # 2. Print Dataset Stats
    print("\n=== Dataset Summary ===")
    print(f"Data path: {path}")
    print(f"Found {len(full_dataset.csv_files)} CSV files")
    print(f"Total sequences: {len(full_dataset)}")
    print(f"Sample graph features: {full_dataset[0][0]}")

    
    # 3. Split Dataset
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_set, test_set = random_split(
        full_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    print(f"\nTrain sequences: {len(train_set)}")
    print(f"Test sequences: {len(test_set)}")
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, 
                            collate_fn=lambda x: x)  # Custom collate for sequences
    test_loader = DataLoader(test_set, batch_size=32, collate_fn=lambda x: x)

    # Model & Optimizer
    model = GCN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(10):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            preds = model(batch)
            labels = torch.stack([seq[-1].y for seq in batch])  # Last frame labels
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
    return model, test_set, test_loader

# --------------------------
# 6. Evaluation
# --------------------------
def evaluate(loader):
    model.eval()
    correct = 0
    for batch in loader:
        preds = model(batch)
        pred_labels = (torch.sigmoid(preds) > 0.5).float()
        true_labels = torch.stack([seq[-1].y for seq in batch])
        correct += (pred_labels == true_labels).all(dim=1).sum().item()
    return correct / len(loader.dataset)

# --------------------------
# 7. Prediction Visualization
# --------------------------
def visualize_prediction(model, sample):
    model.eval()
    with torch.no_grad():
        pred = torch.sigmoid(model([sample])[0])
    
    print(f"\nSample Prediction:")
    print(f"  Blue Team Goal Probability:  {pred[0]:.1%}")
    print(f"  Orange Team Goal Probability: {pred[1]:.1%}")
    print(f"  Actual Labels: Blue={sample.y[0]}, Orange={sample.y[1]}")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize W&B
    wandb.init(project="rl-goal-prediction")
    
    # Train and evaluate
    model, test_set, test_loader = train()
    print(f"Test Accuracy: {evaluate(test_loader, model):.2%}")
    sample = test_set[0][-1]
    visualize_prediction(model, sample)
    
    # Save model
    torch.save(model.state_dict(), "model.pth")
    wandb.save("model.pth")