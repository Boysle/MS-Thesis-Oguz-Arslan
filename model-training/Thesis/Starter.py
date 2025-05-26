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

# ====================== CONFIGURATION ======================
DEBUG = True  # Set to False to reduce print output
VISUALIZE = True
NUM_PLAYERS = 6
PLAYER_FEATURES = 3  # x,y,z coordinates
HIDDEN_DIM = 32
BATCH_SIZE = 32
EPOCHS = 30
TEST_SIZE = 0.2
RANDOM_SEED = 42

# ====================== DATA PROCESSING ======================
def load_and_process_data(csv_path):
    """Load single CSV file and convert to PyG Data objects"""
    if DEBUG:
        print(f"\n=== LOADING DATA FROM {csv_path} ===")
        print("Expected CSV format:")
        print("p0_pos_x, p0_pos_y, p0_pos_z, p0_team, ..., p5_pos_x, p5_pos_y, p5_pos_z, p5_team, team_0_goal_prev_5s, team_1_goal_prev_5s")
    
    df = pd.read_csv(csv_path)
    if DEBUG:
        print(f"\nRaw CSV head:\n{df.head()}")
        print(f"\nData shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

    dataset = []
    for idx, row in df.iterrows():
        # Extract node features (positions and teams)
        x = []
        team_labels = []
        for i in range(NUM_PLAYERS):
            pos = [row[f'p{i}_pos_x'], row[f'p{i}_pos_y'], row[f'p{i}_pos_z']]
            team = row[f'p{i}_team']
            x.append(pos)
            team_labels.append(team)
        
        x = torch.tensor(x, dtype=torch.float)
        team_labels = torch.tensor(team_labels, dtype=torch.long)
        
        # Calculate edge weights (inverse distances)
        edge_index = []
        edge_weights = []
        for i in range(NUM_PLAYERS):
            for j in range(NUM_PLAYERS):
                if i != j:
                    distance = torch.norm(x[i] - x[j])
                    edge_index.append([i, j])
                    edge_weights.append(1.0 / (1.0 + distance))
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        
        # Get labels
        orange_y = torch.tensor([row['team_0_goal_prev_5s']], dtype=torch.long)
        blue_y = torch.tensor([row['team_1_goal_prev_5s']], dtype=torch.long)
        
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weights,
            orange_y=orange_y,
            blue_y=blue_y,
            team_labels=team_labels,
            game_id=idx
        )
        dataset.append(data)
        
        if DEBUG and idx < 2:  # Print first 2 samples for verification
            print(f"\n=== Sample {idx} ===")
            print("Node positions:\n", x)
            print("Team labels (0=orange, 1=blue):", team_labels)
            print("Edge index shape:", edge_index.shape)
            print("Edge weights (first 5):", edge_weights[:5])
            print("Labels - Orange:", orange_y.item(), "Blue:", blue_y.item())
    
    return dataset

# ====================== VISUALIZATION ======================
def visualize_game_state(data, title=None):
    """Visualize a single game state with team colors and edge weights"""
    G = to_networkx(data, to_undirected=True)
    pos = {i: data.x[i,:2].tolist() for i in range(NUM_PLAYERS)}  # Use x,y positions
    
    plt.figure(figsize=(10, 8))
    
    # Node colors based on team (0=orange, 1=blue)
    node_colors = ['orange' if data.team_labels[i] == 0 else 'blue' 
                  for i in range(NUM_PLAYERS)]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500)
    
    # Edge widths proportional to weights
    edge_widths = [data.edge_weight[i].item()*3 for i in range(data.edge_index.size(1))]
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.4)
    
    # Add team labels
    for i in range(NUM_PLAYERS):
        plt.text(pos[i][0], pos[i][1], f"P{i}", 
                 bbox=dict(facecolor='white', alpha=0.7))
    
    plt.title(title or f"Game {data.game_id}\n"
              f"Orange Score: {data.orange_y.item()} | Blue Score: {data.blue_y.item()}")
    plt.grid(True)
    plt.show()

def plot_embeddings(model, loader, epoch):
    """Visualize learned node embeddings using t-SNE"""
    model.eval()
    embeddings, colors = [], []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # Get embeddings before prediction heads
            x = model.conv1(batch.x, batch.edge_index, batch.edge_weight)
            x = model.conv2(x, batch.edge_index, batch.edge_weight)
            
            embeddings.append(x.cpu())
            colors.extend(['orange' if t == 0 else 'blue' 
                         for t in batch.team_labels.cpu().numpy()])
    
    embeddings = torch.cat(embeddings).numpy()
    
    # Reduce dimensionality
    tsne = TSNE(n_components=2, perplexity=30)
    emb_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    for color in ['orange', 'blue']:
        mask = np.array(colors) == color
        plt.scatter(emb_2d[mask,0], emb_2d[mask,1], c=color, 
                   label=f'{color} team', alpha=0.6)
    plt.title(f'Epoch {epoch} - Learned Node Embeddings')
    plt.legend()
    plt.show()

# ====================== MODEL ARCHITECTURE ======================
class RocketLeagueGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(PLAYER_FEATURES, HIDDEN_DIM)
        self.conv2 = GCNConv(HIDDEN_DIM, HIDDEN_DIM)
        self.orange_head = nn.Linear(HIDDEN_DIM, 1)
        self.blue_head = nn.Linear(HIDDEN_DIM, 1)

        if DEBUG:
            print("\n=== MODEL ARCHITECTURE ===")
            print(self)
            print(f"Input dim: {PLAYER_FEATURES} (xyz coordinates)")
            print(f"Hidden dim: {HIDDEN_DIM}")
            print("Edge weights: Inverse distance (1/(1+d))")

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
        
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
        x = global_mean_pool(x, batch)
        
        # Direct sigmoid application
        return torch.sigmoid(self.orange_head(x)), torch.sigmoid(self.blue_head(x))


# ====================== MAIN EXECUTION ======================
if __name__ == "__main__":
    # Load and process data local pc and hpc case
    # dataset = load_and_process_data('E:\\Raw RL Esports Replays\\Day 3 Swiss Stage\\starter_all_replays_combined.csv')  # Change to your CSV path
    dataset = load_and_process_data('/home/oguz.arslan1/datasets/starter_all_replays_combined.csv')

    # Train-test split
    train_data, test_data = train_test_split(
        dataset, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    
    if DEBUG:
        print(f"\n=== DATA SPLIT ===")
        print(f"Total samples: {len(dataset)}")
        print(f"Train samples: {len(train_data)}")
        print(f"Test samples: {len(test_data)}")
        print(f"Batch size: {BATCH_SIZE}")
        
        # Visualize first 2 training samples
        if VISUALIZE:
            print("\n=== VISUALIZING TRAINING SAMPLES ===")
            for i in range(2):
                visualize_game_state(
                    train_data[i],
                    f"Training Sample {i}\n"
                    f"Orange: {train_data[i].orange_y.item()}, "
                    f"Blue: {train_data[i].blue_y.item()}"
                )

    # Initialize model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RocketLeagueGCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    # Training loop
    print("\n=== TRAINING STARTED ===")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct_orange = 0
        correct_blue = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            orange_pred, blue_pred = model(batch)
            
            # Calculate losses
            loss_orange = criterion(orange_pred, batch.orange_y.float().unsqueeze(1))
            loss_blue = criterion(blue_pred, batch.blue_y.float().unsqueeze(1))
            loss = loss_orange + loss_blue
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            correct_orange += ((orange_pred > 0.5) == batch.orange_y.unsqueeze(1)).sum().item()
            correct_blue += ((blue_pred > 0.5) == batch.blue_y.unsqueeze(1)).sum().item()
        
        # Validation
        model.eval()
        test_correct_orange = 0
        test_correct_blue = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                orange_pred, blue_pred = model(batch)
                test_correct_orange += ((orange_pred > 0.5) == batch.orange_y.unsqueeze(1)).sum().item()
                test_correct_blue += ((blue_pred > 0.5) == batch.blue_y.unsqueeze(1)).sum().item()
        
        # Print epoch stats
        train_acc_orange = correct_orange / len(train_data)
        train_acc_blue = correct_blue / len(train_data)
        test_acc_orange = test_correct_orange / len(test_data)
        test_acc_blue = test_correct_blue / len(test_data)
        
        print(f"Epoch {epoch:02d} | Loss: {total_loss/len(train_loader):.4f} | "
              f"Train Acc: O {train_acc_orange:.3f}, B {train_acc_blue:.3f} | "
              f"Test Acc: O {test_acc_orange:.3f}, B {test_acc_blue:.3f}")
        
        # Visualizations
        if VISUALIZE and (epoch % 5 == 0 or epoch == EPOCHS-1):
            plot_embeddings(model, train_loader, epoch)
            
            # Show test predictions
            model.eval()
            with torch.no_grad():
                test_sample = test_data[0].to(device)
                orange_prob, blue_prob = model(test_sample)
                visualize_game_state(
                    test_data[0],
                    f"Test Sample Prediction\n"
                    f"True: O {test_data[0].orange_y.item()}, B {test_data[0].blue_y.item()}\n"
                    f"Pred: O {orange_prob.item():.2f}, B {blue_prob.item():.2f}"
                )

    # Final evaluation
    def evaluate(loader):
        model.eval()
        correct_orange, correct_blue = 0, 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                orange_pred, blue_pred = model(batch)
                correct_orange += ((orange_pred > 0.5) == batch.orange_y.unsqueeze(1)).sum().item()
                correct_blue += ((blue_pred > 0.5) == batch.blue_y.unsqueeze(1)).sum().item()
        return correct_orange/len(loader.dataset), correct_blue/len(loader.dataset)

    train_orange, train_blue = evaluate(train_loader)
    test_orange, test_blue = evaluate(test_loader)

    print("\n=== FINAL RESULTS ===")
    print(f"Train Accuracy - Orange: {train_orange:.4f}, Blue: {train_blue:.4f}")
    print(f"Test Accuracy - Orange: {test_orange:.4f}, Blue: {test_blue:.4f}")