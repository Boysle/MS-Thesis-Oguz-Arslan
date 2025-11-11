import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import linecache
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader  # Use standard PyTorch DataLoader
from torch_geometric.nn import GATConv, global_mean_pool, GlobalAttention
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, precision_score, recall_score, precision_recall_curve, 
    confusion_matrix, accuracy_score, average_precision_score, log_loss
)

# ====================== CONFIGURATION & CONSTANTS ======================
# --- 6-Node (Ball-less) Architecture ---
NUM_NODES_6 = 6
NODE_FEATURES_6 = 13 # pos, vel, fwd, boost, team, alive, dist_to_ball
GLOBAL_FEATURES_6 = 14 # ball_pos, ball_vel, 6 pads, hit_team, time

# --- 7-Node (Ball-Added) Architecture ---
NUM_NODES_7 = 7
NODE_FEATURES_7 = 12 # pos, vel, fwd, boost, team, alive
GLOBAL_FEATURES_7 = 8  # 6 pads, hit_team, time

# --- Normalization (Universal) ---
POS_MIN_X, POS_MAX_X = -4096, 4096; POS_MIN_Y, POS_MAX_Y = -6000, 6000; POS_MIN_Z, POS_MAX_Z = 0, 2044
VEL_MIN, VEL_MAX = -2300, 2300; BOOST_MIN, BOOST_MAX = 0, 100; BALL_VEL_MIN, BALL_VEL_MAX = -6000, 6000
BOOST_PAD_MIN, BOOST_PAD_MAX = 0, 10; DIST_MIN, DIST_MAX = 0, (8192**2 + 10240**2 + 2044**2)**0.5

def normalize(val, min_val, max_val):
    return (val - min_val) / (max_val - min_val + 1e-8)

# ====================== MODEL & DATA CLASSES (MATCHING TRAIN SCRIPTS) ======================
class RocketLeagueGAT(nn.Module):
    def __init__(self, player_features, global_features, hidden_dim, edge_dim):
        super().__init__()
        self.conv1 = GATConv(player_features, hidden_dim, heads=4, edge_dim=edge_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 4)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, edge_dim=edge_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # 1. Define the "gate" network for the attention. 
        #    It's a simple MLP that learns the importance score.
        self.gate_nn = nn.Linear(hidden_dim, 1) 
        
        # 2. Define the GlobalAttention layer
        self.pool = GlobalAttention(gate_nn=self.gate_nn)

        # The rest of the model remains the same
        self.orange_head = nn.Sequential(nn.Linear(hidden_dim + global_features, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))
        self.blue_head = nn.Sequential(nn.Linear(hidden_dim + global_features, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_attr)))
        
        # 3. Use the new attention pooling layer.
        #    It takes the node features (x) and the batch index.
        graph_embed = self.pool(x, data.batch) 
        
        # The rest of the forward pass is identical
        combined = torch.cat([graph_embed, data.global_features], dim=1)
        return self.orange_head(combined), self.blue_head(combined)

class GraphLazyDataset(torch.utils.data.Dataset):
    # This Dataset is flexible for 6/7 nodes AND edge_attr
    def __init__(self, list_of_csv_paths, model_arch='6node', edge_feature_dim=4):
        self.csv_paths = list_of_csv_paths
        self.model_arch = model_arch
        self.edge_feature_dim = edge_feature_dim
        self.file_info, self.cumulative_rows, self.header, total_rows = [], [0], None, 0

        if self.model_arch == '6node':
            self.num_nodes = NUM_NODES_6
            self.node_features = NODE_FEATURES_6
            self.global_features = GLOBAL_FEATURES_6
        elif self.model_arch == '7node':
            self.num_nodes = NUM_NODES_7
            self.node_features = NODE_FEATURES_7
            self.global_features = GLOBAL_FEATURES_7
        else:
            raise ValueError("model_arch must be '6node' or '7node'")

        if not list_of_csv_paths: self.length = 0; return
        desc = f"Indexing {os.path.basename(os.path.dirname(list_of_csv_paths[0]))} files"
        for path in tqdm(self.csv_paths, desc=desc):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    if self.header is None: self.header = f.readline().strip().split(',')
                    num_lines = sum(1 for _ in f)
                if num_lines > 0: self.file_info.append({'path': path, 'rows': num_lines}); total_rows += num_lines; self.cumulative_rows.append(total_rows)
            except Exception as e: print(f"\nWarning: Could not process file {path}. Skipping. Error: {e}")
        
        self.length = total_rows
        self.edge_index = torch.tensor([(i, j) for i in range(self.num_nodes) for j in range(self.num_nodes) if i != j], dtype=torch.long).t().contiguous()
        self.skipped_count = 0
        print(f"\n--- Indexing complete. Total samples: {self.length}. Arch: {self.model_arch}. Edge Feats: {self.edge_feature_dim} ---")

    def __len__(self): return self.length

    def normalized_dot_product(self, v1, v2):
        v1_norm = v1 / (torch.linalg.norm(v1, dim=-1, keepdim=True) + 1e-8)
        v2_norm = v2 / (torch.linalg.norm(v2, dim=-1, keepdim=True) + 1e-8)
        dot = torch.sum(v1_norm * v2_norm, dim=-1)
        return (dot + 1) / 2

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length: return None
        file_index = np.searchsorted(self.cumulative_rows, idx, side='right') - 1
        file_path = self.file_info[file_index]['path']; local_idx = idx - self.cumulative_rows[file_index]
        line = linecache.getline(file_path, local_idx + 2)
        if not line.strip(): self.skipped_count += 1; return None
        
        try:
            row = dict(zip(self.header, line.strip().split(',')))
            x_features = []
            
            if self.model_arch == '6node':
                # --- 6-Node (Ball-less) Feature Building ---
                for i in range(self.num_nodes): # 6 players
                    x_features.append([
                        normalize(float(row[f'p{i}_pos_x']), POS_MIN_X, POS_MAX_X),
                        normalize(float(row[f'p{i}_pos_y']), POS_MIN_Y, POS_MAX_Y),
                        normalize(float(row[f'p{i}_pos_z']), POS_MIN_Z, POS_MAX_Z),
                        normalize(float(row[f'p{i}_vel_x']), VEL_MIN, VEL_MAX),
                        normalize(float(row[f'p{i}_vel_y']), VEL_MIN, VEL_MAX),
                        normalize(float(row[f'p{i}_vel_z']), VEL_MIN, VEL_MAX),
                        float(row[f'p{i}_forward_x']),
                        float(row[f'p{i}_forward_y']),
                        float(row[f'p{i}_forward_z']),
                        normalize(float(row[f'p{i}_boost_amount']), BOOST_MIN, BOOST_MAX),
                        float(row[f'p{i}_team']),
                        float(row[f'p{i}_alive']),
                        normalize(float(row[f'p{i}_dist_to_ball']), DIST_MIN, DIST_MAX)
                    ])
                global_features = [
                    normalize(float(row['ball_pos_x']), POS_MIN_X, POS_MAX_X),
                    normalize(float(row['ball_pos_y']), POS_MIN_Y, POS_MAX_Y),
                    normalize(float(row['ball_pos_z']), POS_MIN_Z, POS_MAX_Z),
                    normalize(float(row['ball_vel_x']), BALL_VEL_MIN, BALL_VEL_MAX),
                    normalize(float(row['ball_vel_y']), BALL_VEL_MIN, BALL_VEL_MAX),
                    normalize(float(row['ball_vel_z']), BALL_VEL_MIN, BALL_VEL_MAX),
                    normalize(float(row['boost_pad_0_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX),
                    normalize(float(row['boost_pad_1_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX),
                    normalize(float(row['boost_pad_2_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX),
                    normalize(float(row['boost_pad_3_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX),
                    normalize(float(row['boost_pad_4_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX),
                    normalize(float(row['boost_pad_5_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX),
                    float(row['ball_hit_team_num']),
                    normalize(min(float(row['seconds_remaining']), 300.0), 0, 300)
                ]
            
            elif self.model_arch == '7node':
                # --- 7-Node (Ball-Added) Feature Building ---
                for i in range(self.num_nodes - 1): # 6 players
                    x_features.append([
                        normalize(float(row[f'p{i}_pos_x']), POS_MIN_X, POS_MAX_X),
                        normalize(float(row[f'p{i}_pos_y']), POS_MIN_Y, POS_MAX_Y),
                        normalize(float(row[f'p{i}_pos_z']), POS_MIN_Z, POS_MAX_Z),
                        normalize(float(row[f'p{i}_vel_x']), VEL_MIN, VEL_MAX),
                        normalize(float(row[f'p{i}_vel_y']), VEL_MIN, VEL_MAX),
                        normalize(float(row[f'p{i}_vel_z']), VEL_MIN, VEL_MAX),
                        float(row[f'p{i}_forward_x']),
                        float(row[f'p{i}_forward_y']),
                        float(row[f'p{i}_forward_z']),
                        normalize(float(row[f'p{i}_boost_amount']), BOOST_MIN, BOOST_MAX),
                        float(row[f'p{i}_team']),
                        float(row[f'p{i}_alive']),
                    ])
                # Node 6: Ball
                x_features.append([
                    normalize(float(row['ball_pos_x']), POS_MIN_X, POS_MAX_X),
                    normalize(float(row['ball_pos_y']), POS_MIN_Y, POS_MAX_Y),
                    normalize(float(row['ball_pos_z']), POS_MIN_Z, POS_MAX_Z),
                    normalize(float(row['ball_vel_x']), BALL_VEL_MIN, BALL_VEL_MAX),
                    normalize(float(row['ball_vel_y']), BALL_VEL_MIN, BALL_VEL_MAX),
                    normalize(float(row['ball_vel_z']), BALL_VEL_MIN, BALL_VEL_MAX),
                    0.0, 0.0, 0.0, # Padding for fwd
                    0.0, # Padding for boost
                    2.0, # Padding for team
                    1.0, # Padding for alive
                ])
                # New Global Features
                global_features = [
                    normalize(float(row['boost_pad_0_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX),
                    normalize(float(row['boost_pad_1_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX),
                    normalize(float(row['boost_pad_2_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX),
                    normalize(float(row['boost_pad_3_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX),
                    normalize(float(row['boost_pad_4_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX),
                    normalize(float(row['boost_pad_5_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX),
                    float(row['ball_hit_team_num']),
                    normalize(min(float(row['seconds_remaining']), 300.0), 0, 300)
                ]

            x_tensor = torch.tensor(x_features, dtype=torch.float32)
            global_tensor = torch.tensor(global_features, dtype=torch.float32).unsqueeze(0)
            
            # --- Build Edge Attributes ---
            positions = x_tensor[:, 0:3]
            velocities = x_tensor[:, 3:6]
            forwards = x_tensor[:, 6:9]
            teams = x_tensor[:, 10]
            
            edge_attrs = []
            for i, j in self.edge_index.t():
                dist = torch.linalg.norm(positions[i] - positions[j])
                inv_dist = 1.0 / (1.0 + (dist / 1500.0)**2.0)
                same_team = 1.0 if teams[i] == teams[j] else 0.0
                vec_i_to_j = positions[j] - positions[i]
                vel_mag = torch.linalg.norm(velocities[i])
                scaled_vel_mag = vel_mag / (2300.0 + 1e-8)
                vel_align = self.normalized_dot_product(velocities[i], vec_i_to_j)
                velocity_feature = vel_align * scaled_vel_mag
                forward_align = self.normalized_dot_product(forwards[i], vec_i_to_j)
                
                all_features = [inv_dist, same_team, velocity_feature, forward_align]
                edge_attrs.append(all_features[:self.edge_feature_dim])

            edge_attr_tensor = torch.tensor(edge_attrs, dtype=torch.float32)

            # Labels
            orange_y = torch.tensor([float(row['team_1_goal_in_event_window'])], dtype=torch.float32)
            blue_y = torch.tensor([float(row['team_0_goal_in_event_window'])], dtype=torch.float32)
            
            return Data(x=x_tensor, edge_index=self.edge_index, edge_attr=edge_attr_tensor,
                        global_features=global_tensor, y_orange=orange_y, y_blue=blue_y)
        
        except (ValueError, KeyError, IndexError): 
            self.skipped_count += 1
            return None

    def empty_data(self):
        x_tensor = torch.zeros((self.num_nodes, self.node_features), dtype=torch.float32)
        global_tensor = torch.zeros((1, self.global_features), dtype=torch.float32)
        edge_attr_tensor = torch.zeros((self.edge_index.size(1), self.edge_feature_dim), dtype=torch.float32)
        return Data(x=x_tensor, edge_index=self.edge_index, edge_attr=edge_attr_tensor,
                    global_features=global_tensor,
                    y_orange=torch.tensor([0.0]), y_blue=torch.tensor([0.0]))

def collate_fn_master(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return Batch.from_data_list(batch)

# ====================== HELPER FUNCTIONS ======================

def get_predictions_and_loss(model, dataloader, device, pos_weight_orange, pos_weight_blue):
    model.eval()
    criterion_o = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_orange]).to(device))
    criterion_b = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_blue]).to(device))
    
    all_orange_labels, all_blue_labels = [], []
    all_orange_probs, all_blue_probs = [], []
    total_loss_o, total_loss_b = 0.0, 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Getting predictions and loss"):
            if batch is None: continue
            batch = batch.to(device)
            
            orange_logits, blue_logits = model(batch)
            
            total_loss_o += criterion_o(orange_logits, batch.y_orange.view_as(orange_logits)).item() * batch.num_graphs
            total_loss_b += criterion_b(blue_logits, batch.y_blue.view_as(blue_logits)).item() * batch.num_graphs
            
            all_orange_labels.extend(batch.y_orange.cpu().numpy().flatten())
            all_blue_labels.extend(batch.y_blue.cpu().numpy().flatten())
            all_orange_probs.extend(torch.sigmoid(orange_logits).cpu().numpy().flatten())
            all_blue_probs.extend(torch.sigmoid(blue_logits).cpu().numpy().flatten())
            
    num_samples = len(all_orange_labels)
    avg_loss_o = total_loss_o / num_samples if num_samples > 0 else 0
    avg_loss_b = total_loss_b / num_samples if num_samples > 0 else 0
            
    return (np.array(all_orange_labels), np.array(all_blue_labels), 
            np.array(all_orange_probs), np.array(all_blue_probs),
            avg_loss_o, avg_loss_b)

def plot_and_save_distribution(y_true, y_pred_proba, threshold, model_type, team_name, set_name, output_dir):
    plt.figure(figsize=(12, 8)); 
    sns.histplot(y_pred_proba[y_true == 0], bins=50, color='skyblue', label='Actual: No Goal', kde=True, log_scale=(False, True)); 
    sns.histplot(y_pred_proba[y_true == 1], bins=50, color='salmon', label='Actual: Goal', kde=True, log_scale=(False, True)); 
    plt.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Default Threshold (0.5)'); 
    plt.axvline(threshold, color='red', linestyle='-', linewidth=2, label=f'Custom Threshold ({threshold:.4f})'); 
    plt.grid(True, which='both', linestyle='--', linewidth=0.5); 
    plt.title(f'Log-Scale Prob. Distribution on {set_name} Set ({model_type.upper()} - {team_name.upper()})', fontsize=16); 
    plt.xlabel('Predicted Probability of Goal'); plt.ylabel('Frequency (Log Scale)'); plt.legend(); 
    filepath = os.path.join(output_dir, f'{team_name}_{set_name.lower()}_distribution.png'); 
    plt.savefig(filepath); plt.close(); 
    print(f"  Distribution plot saved to: {filepath}")

def plot_and_save_confusion_matrix(y_true, preds, threshold, model_type, team_name, set_name, output_dir):
    cm = confusion_matrix(y_true, preds, labels=[0,1]); 
    plt.figure(figsize=(8, 6)); 
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Goal", "Goal"], yticklabels=["No Goal", "Goal"]); 
    plt.title(f'CM on {set_name} Set ({model_type.upper()} - {team_name.upper()}) at Threshold {threshold:.4f}', fontsize=14); 
    plt.ylabel('Actual Label'); plt.xlabel('Predicted Label'); 
    thresh_str = str(round(threshold, 4)).replace('.', 'p'); 
    filepath = os.path.join(output_dir, f'{team_name}_{set_name.lower()}_cm_thresh_{thresh_str}.png'); 
    plt.savefig(filepath, bbox_inches='tight'); plt.close(); 
    print(f"  Confusion matrix plot saved to: {filepath}")

def plot_and_save_metrics_table(metrics_default, metrics_optimized, threshold, model_type, team_name, set_name, output_dir):
    metrics_data = [[f"{metrics_default[0]:.4f}", f"{metrics_optimized[0]:.4f}"], [f"{metrics_default[1]:.4f}", f"{metrics_optimized[1]:.4f}"], [f"{metrics_default[2]:.4f}", f"{metrics_optimized[2]:.4f}"]]
    fig, ax = plt.subplots(figsize=(10, 2)); ax.axis('tight'); ax.axis('off'); 
    table = ax.table(cellText=metrics_data, rowLabels=['Precision', 'Recall', 'F1-Score'], colLabels=[f'Default (0.5)', f'Custom ({threshold:.4f})'], cellLoc='center', loc='center', colWidths=[0.3, 0.3]); 
    table.auto_set_font_size(False); table.set_fontsize(12); table.scale(1.2, 1.2); 
    plt.title(f'Performance Metrics on {set_name} Set ({model_type.upper()} - {team_name.upper()})', fontsize=14, pad=20); 
    filepath = os.path.join(output_dir, f'{team_name}_{set_name.lower()}_metrics_table.png'); 
    plt.savefig(filepath, bbox_inches='tight', dpi=150); plt.close(); 
    print(f"  Metrics table plot saved to: {filepath}")

# ====================== MAIN EXECUTION ======================
def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Full Analysis for GAT Model")
    
    parser.add_argument('--model-path', type=str, required=True, help='Path to the saved GAT model checkpoint (.pth file).')
    parser.add_argument('--model-arch', type=str, required=True, choices=['6node', '7node'], help='Architecture of the GAT (6node or 7node).')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to the parent directory of train/val/test splits.')
    parser.add_argument('--team', type=str, required=True, choices=['orange', 'blue'], help='Which team to analyze.')
    parser.add_argument('--threshold', type=float, required=True, help='The custom threshold to use for analysis.')
    
    # Add edge-features arg, but try to read from checkpoint first
    parser.add_argument('--edge-features', type=int, default=4, choices=[1, 2, 3, 4], help='Number of edge features to use (1-4 for ablation). Fallback if not in checkpoint.')
    
    parser.add_argument('--output-dir', type=str, default='./analysis_results_gat', help='Directory to save all output figures.')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size for evaluation.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA evaluation.')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of workers for DataLoader.')

    args = parser.parse_args()
    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")
    
    # Create output directory
    output_dir_team = os.path.join(args.output_dir, f"{args.model_arch}_{args.team}_e{args.edge_features}")
    os.makedirs(output_dir_team, exist_ok=True)
    print(f"--- Saving outputs to: {output_dir_team} ---")

    # --- 1. Calculate Class Weights (Needed for Loss) ---
    print("\n--- Calculating class weights from training data ---")
    train_dir = os.path.join(args.data_dir, 'train')
    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.csv')]
    if not train_files:
        print(f"CRITICAL ERROR: No training files found in {train_dir}. Cannot calculate weights."); return

    pos_orange, neg_orange, pos_blue, neg_blue = 0, 0, 0, 0
    for file in tqdm(train_files, desc="Scanning labels"):
        try:
            df = pd.read_csv(file, usecols=['team_1_goal_in_event_window', 'team_0_goal_in_event_window'])
            pos_orange += df['team_1_goal_in_event_window'].sum(); neg_orange += len(df) - df['team_1_goal_in_event_window'].sum()
            pos_blue += df['team_0_goal_in_event_window'].sum(); neg_blue += len(df) - df['team_0_goal_in_event_window'].sum()
        except Exception as e:
            print(f"Warning: Skipping file {file} due to error: {e}")
    
    pos_weight_orange = (neg_orange / pos_orange) if pos_orange > 0 else 1.0
    pos_weight_blue = (neg_blue / pos_blue) if pos_blue > 0 else 1.0
    print(f"Positional weight for Orange loss: {pos_weight_orange:.2f}")
    print(f"Positional weight for Blue loss: {pos_weight_blue:.2f}")

    # --- 2. Load Data ---
    print(f"\n--- Loading VAL and TEST data (arch: {args.model_arch}, edge_feats: {args.edge_features}) ---")
    val_dir = os.path.join(args.data_dir, 'val'); test_dir = os.path.join(args.data_dir, 'test')
    val_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.csv')]
    test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.csv')]
    
    val_dataset = GraphLazyDataset(val_files, model_arch=args.model_arch, edge_feature_dim=args.edge_features)
    test_dataset = GraphLazyDataset(test_files, model_arch=args.model_arch, edge_feature_dim=args.edge_features)
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn_master)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn_master)

    # --- 3. Load Model ---
    print(f"\n--- Loading model from {args.model_path} ---")
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        model_args = checkpoint.get('args', {})
        
        # Read parameters from checkpoint, falling back to args
        hidden_dim = model_args.get('hidden_dim', 64) 
        edge_dim = model_args.get('edge_features', args.edge_features) # <-- Important!
        
        if edge_dim != args.edge_features:
             print(f"Warning: Model checkpoint was trained with edge_features={edge_dim}, but you passed --edge-features={args.edge_features}. Using {edge_dim} from checkpoint.")
        
        if args.model_arch == '6node':
            node_features, global_features = NODE_FEATURES_6, GLOBAL_FEATURES_6
        else: # 7node
            node_features, global_features = NODE_FEATURES_7, GLOBAL_FEATURES_7
            
        print(f"Initializing GAT with: NodeFeat={node_features}, GlobalFeat={global_features}, HiddenDim={hidden_dim}, EdgeDim={edge_dim}")
        model = RocketLeagueGAT(node_features, global_features, hidden_dim, edge_dim).to(device)
        
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load the model. Ensure path, arch, and edge-features match. Error: {e}"); return

    # --- 4. Run Analysis on VALIDATION Set ---
    print(f"\n--- Step 1: Running Analysis on VALIDATION set for {args.team.upper()} team ---")
    val_labels_o, val_labels_b, val_probs_o, val_probs_b, val_loss_o, val_loss_b = get_predictions_and_loss(
        model, val_loader, device, pos_weight_orange, pos_weight_blue)
    
    y_true_val = val_labels_o if args.team == 'orange' else val_labels_b
    y_prob_val = val_probs_o if args.team == 'orange' else val_probs_b
    val_loss = val_loss_o if args.team == 'orange' else val_loss_b
    
    preds_def_val = (y_prob_val > 0.5).astype(int)
    preds_opt_val = (y_prob_val > args.threshold).astype(int)
    
    print("\n--- FINAL VALIDATION RESULTS ---")
    print(f"  {args.team.upper()} Team Weighted Log Loss: {val_loss:.4f}")
    
    tn, fp, fn, tp = confusion_matrix(y_true_val, preds_def_val, labels=[0,1]).ravel()
    print(f"\n-- Default @ 0.5 Threshold --")
    print(f"  F1: {f1_score(y_true_val, preds_def_val, zero_division=0):.4f} | P: {precision_score(y_true_val, preds_def_val, zero_division=0):.4f} | R: {recall_score(y_true_val, preds_def_val, zero_division=0):.4f} | Acc: {accuracy_score(y_true_val, preds_def_val):.4f}")
    print(f"    -> TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")
    
    tn, fp, fn, tp = confusion_matrix(y_true_val, preds_opt_val, labels=[0,1]).ravel()
    print(f"\n-- Custom @ {args.threshold} Threshold --")
    print(f"  F1: {f1_score(y_true_val, preds_opt_val, zero_division=0):.4f} | P: {precision_score(y_true_val, preds_opt_val, zero_division=0):.4f} | R: {recall_score(y_true_val, preds_opt_val, zero_division=0):.4f} | Acc: {accuracy_score(y_true_val, preds_opt_val):.4f}")
    print(f"    -> TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")
    
    print(f"\n-- Threshold-Independent --")
    print(f"  AUPRC: {average_precision_score(y_true_val, y_prob_val):.4f}")
    
    # --- 5. Run Analysis on TEST Set ---
    print(f"\n--- Step 2: Running Analysis on TEST set for {args.team.upper()} team ---")
    test_labels_o, test_labels_b, test_probs_o, test_probs_b, test_loss_o, test_loss_b = get_predictions_and_loss(
        model, test_loader, device, pos_weight_orange, pos_weight_blue)
    
    y_true_test = test_labels_o if args.team == 'orange' else test_labels_b
    y_prob_test = test_probs_o if args.team == 'orange' else test_probs_b
    test_loss = test_loss_o if args.team == 'orange' else test_loss_b

    preds_def_test = (y_prob_test > 0.5).astype(int)
    preds_opt_test = (y_prob_test > args.threshold).astype(int)

    print("\n--- FINAL TEST RESULTS ---")
    print(f"  {args.team.upper()} Team Weighted Log Loss: {test_loss:.4f}")
    
    tn, fp, fn, tp = confusion_matrix(y_true_test, preds_def_test, labels=[0,1]).ravel()
    print(f"\n-- Default @ 0.5 Threshold --")
    print(f"  F1: {f1_score(y_true_test, preds_def_test, zero_division=0):.4f} | P: {precision_score(y_true_test, preds_def_test, zero_division=0):.4f} | R: {recall_score(y_true_test, preds_def_test, zero_division=0):.4f} | Acc: {accuracy_score(y_true_test, preds_def_test):.4f}")
    print(f"    -> TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")
    
    tn, fp, fn, tp = confusion_matrix(y_true_test, preds_opt_test, labels=[0,1]).ravel()
    print(f"\n-- Custom @ {args.threshold} Threshold --")
    print(f"  F1: {f1_score(y_true_test, preds_opt_test, zero_division=0):.4f} | P: {precision_score(y_true_test, preds_opt_test, zero_division=0):.4f} | R: {recall_score(y_true_test, preds_opt_test, zero_division=0):.4f} | Acc: {accuracy_score(y_true_test, preds_opt_test):.4f}")
    print(f"    -> TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")

    print(f"\n-- Threshold-Independent --")
    print(f"  AUPRC: {average_precision_score(y_true_test, y_prob_test):.4f}")
    
    # --- 6. Generate Figures for TEST Set ---
    print("\n--- Step 3: Generating Figures for TEST set ---")
    model_type_str = f'gat_{args.model_arch}_e{edge_dim}' # Use edge_dim from checkpoint
    
    plot_and_save_distribution(y_true_test, y_prob_test, args.threshold, model_type_str, args.team, "Test", output_dir_team)
    
    plot_and_save_confusion_matrix(y_true_test, preds_def_test, 0.5, model_type_str, args.team, "Test", output_dir_team)
    plot_and_save_confusion_matrix(y_true_test, preds_opt_test, args.threshold, model_type_str, args.team, "Test", output_dir_team)
    
    metrics_def = [
        precision_score(y_true_test, preds_def_test, zero_division=0), 
        recall_score(y_true_test, preds_def_test, zero_division=0), 
        f1_score(y_true_test, preds_def_test, zero_division=0)
    ]
    metrics_opt = [
        precision_score(y_true_test, preds_opt_test, zero_division=0), 
        recall_score(y_true_test, preds_opt_test, zero_division=0), 
        f1_score(y_true_test, preds_opt_test, zero_division=0)
    ]
    plot_and_save_metrics_table(metrics_def, metrics_opt, args.threshold, model_type_str, args.team, "Test", output_dir_team)

    end_time = time.time()
    total_seconds = end_time - start_time
    print(f"\n--- Analysis Complete ---")
    print(f"\n--- Total Run Time: {total_seconds // 3600:.0f}h {(total_seconds % 3600) // 60:.0f}m {total_seconds % 60:.2f}s ---")


if __name__ == '__main__':
    main()