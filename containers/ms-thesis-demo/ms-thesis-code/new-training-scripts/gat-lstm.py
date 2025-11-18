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
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool, GlobalAttention
from sklearn.metrics import (
    f1_score, precision_score, recall_score, precision_recall_curve, 
    confusion_matrix, accuracy_score, average_precision_score, log_loss
)
import wandb

# ====================== CONFIGURATION & CONSTANTS ======================
SEQUENCE_LENGTH = 6 # 5 previous + 1 current

# --- 7-Node (Ball-Added) Architecture ---
NUM_NODES = 7
NODE_FEATURES = 12 # pos(3), vel(3), fwd(3), boost(1), team(1), alive(1)
GLOBAL_FEATURES = 8  # 6 pads, hit_team, time
EDGE_DIM = 2 # Hard-coded as requested: Dist + Team

# --- Normalization (Universal) ---
POS_MIN_X, POS_MAX_X = -4096, 4096; POS_MIN_Y, POS_MAX_Y = -6000, 6000; POS_MIN_Z, POS_MAX_Z = 0, 2044
VEL_MIN, VEL_MAX = -2300, 2300; BOOST_MIN, BOOST_MAX = 0, 100; BALL_VEL_MIN, BALL_VEL_MAX = -6000, 6000
BOOST_PAD_MIN, BOOST_PAD_MAX = 0, 10; DIST_MIN, DIST_MAX = 0, (8192**2 + 10240**2 + 2044**2)**0.5

def normalize(val, min_val, max_val):
    return (val - min_val) / (max_val - min_val + 1e-8)

# ====================== DATASET CLASS (Sequential GNNs w/ Outlier Handling) ======================
class SequentialGraphLazyDataset(Dataset):
    def __init__(self, list_of_csv_paths, sequence_length=6):
        self.csv_paths = list_of_csv_paths
        self.sequence_length = sequence_length
        self.file_info, self.cumulative_rows, self.header, total_rows = [], [0], None, 0
        
        # Set constants
        self.num_nodes = NUM_NODES
        self.node_features = NODE_FEATURES
        self.global_features = GLOBAL_FEATURES
        self.edge_feature_dim = EDGE_DIM

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
        self.start_index = self.sequence_length - 1 
        self.effective_length = self.length - self.start_index
        self.skipped_count = 0
        print(f"\n--- Indexing complete. Total rows: {self.length}. Effective samples: {self.effective_length}. Arch: 7-node GAT. Edge Feats: {self.edge_feature_dim} ---")

    def __len__(self):
        return self.effective_length

    def normalized_dot_product(self, v1, v2):
        v1_norm = v1 / (torch.linalg.norm(v1, dim=-1, keepdim=True) + 1e-8)
        v2_norm = v2 / (torch.linalg.norm(v2, dim=-1, keepdim=True) + 1e-8)
        dot = torch.sum(v1_norm * v2_norm, dim=-1)
        return (dot + 1) / 2

    def _get_row(self, idx):
        if idx < 0 or idx >= self.length: return None, -1 
        file_index = np.searchsorted(self.cumulative_rows, idx, side='right') - 1
        file_path = self.file_info[file_index]['path']
        local_idx = idx - self.cumulative_rows[file_index]
        line = linecache.getline(file_path, local_idx + 2)
        if not line.strip(): return None, file_index
        try:
            row = dict(zip(self.header, line.strip().split(',')))
            return row, file_index
        except (ValueError, KeyError, IndexError): return None, file_index

    def _build_graph_from_row(self, row):
        """
        Helper function to build a single PyG Data object from a row dict.
        This is the CORRECT version.
        """
        try:
            x_features = []
            # --- Nodes 0-5: Players ---
            for i in range(self.num_nodes - 1): # 6 players
                x_features.append([
                    normalize(float(row[f'p{i}_pos_x']), POS_MIN_X, POS_MAX_X), normalize(float(row[f'p{i}_pos_y']), POS_MIN_Y, POS_MAX_Y), normalize(float(row[f'p{i}_pos_z']), POS_MIN_Z, POS_MAX_Z),
                    normalize(float(row[f'p{i}_vel_x']), VEL_MIN, VEL_MAX), normalize(float(row[f'p{i}_vel_y']), VEL_MIN, VEL_MAX), normalize(float(row[f'p{i}_vel_z']), VEL_MIN, VEL_MAX),
                    float(row[f'p{i}_forward_x']), float(row[f'p{i}_forward_y']), float(row[f'p{i}_forward_z']),
                    normalize(float(row[f'p{i}_boost_amount']), BOOST_MIN, BOOST_MAX), float(row[f'p{i}_team']), float(row[f'p{i}_alive']),
                ])
            # --- Node 6: Ball ---
            x_features.append([
                normalize(float(row['ball_pos_x']), POS_MIN_X, POS_MAX_X), normalize(float(row['ball_pos_y']), POS_MIN_Y, POS_MAX_Y), normalize(float(row['ball_pos_z']), POS_MIN_Z, POS_MAX_Z),
                normalize(float(row['ball_vel_x']), BALL_VEL_MIN, BALL_VEL_MAX), normalize(float(row['ball_vel_y']), BALL_VEL_MIN, BALL_VEL_MAX), normalize(float(row['ball_vel_z']), BALL_VEL_MIN, BALL_VEL_MAX),
                0.0, 0.0, 0.0, 0.0, 2.0, 1.0, # Padding
            ])
            x_tensor = torch.tensor(x_features, dtype=torch.float32)
            
            # --- Global Features ---
            global_features = [
                normalize(float(row['boost_pad_0_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX), normalize(float(row['boost_pad_1_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX), normalize(float(row['boost_pad_2_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX),
                normalize(float(row['boost_pad_3_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX), normalize(float(row['boost_pad_4_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX), normalize(float(row['boost_pad_5_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX),
                float(row['ball_hit_team_num']), normalize(min(float(row['seconds_remaining']), 300.0), 0, 300)
            ]
            global_tensor = torch.tensor(global_features, dtype=torch.float32).unsqueeze(0)
            
            # --- Build Edge Attributes ---
            positions = x_tensor[:, 0:3]; velocities = x_tensor[:, 3:6]; forwards = x_tensor[:, 6:9]; teams = x_tensor[:, 10]
            edge_attrs = []
            for i, j in self.edge_index.t():
                dist = torch.linalg.norm(positions[i] - positions[j]); inv_dist = 1.0 / (1.0 + (dist / 1500.0)**2.0)
                same_team = 1.0 if teams[i] == teams[j] else 0.0
                vec_i_to_j = positions[j] - positions[i]; vel_mag = torch.linalg.norm(velocities[i]); scaled_vel_mag = vel_mag / (2300.0 + 1e-8)
                vel_align = self.normalized_dot_product(velocities[i], vec_i_to_j); velocity_feature = vel_align * scaled_vel_mag
                forward_align = self.normalized_dot_product(forwards[i], vec_i_to_j)
                all_features = [inv_dist, same_team, velocity_feature, forward_align]
                edge_attrs.append(all_features[:self.edge_feature_dim]) # Slices to 2 features
            
            edge_attr_tensor = torch.tensor(edge_attrs, dtype=torch.float32)
            orange_y = torch.tensor([float(row['team_1_goal_in_event_window'])], dtype=torch.float32)
            blue_y = torch.tensor([float(row['team_0_goal_in_event_window'])], dtype=torch.float32)
            
            return Data(x=x_tensor, edge_index=self.edge_index, edge_attr=edge_attr_tensor,
                        global_features=global_tensor, y_orange=orange_y, y_blue=blue_y)
        except Exception as e:
            # You can print the error if you want, but 'None' is cleaner
            # print(f"Error in _build_graph_from_row: {e}")
            return None

    def __getitem__(self, idx):
        current_idx = idx + self.start_index
        sequence_data = []
        outlier_count_replay, outlier_count_score, outlier_count_overtime = 0, 0, 0
        
        anchor_row, anchor_file_idx = self._get_row(current_idx)
        if anchor_row is None: self.skipped_count += 1; return None
        
        anchor_replay_id = anchor_row['replay_id']
        anchor_score_diff = anchor_row['score_difference']
        anchor_ball_hit = anchor_row.get('ball_hit_team_num', '0.5')
        
        sequence_rows.append(anchor_row)
        last_valid_row = anchor_row

        for k in range(1, self.sequence_length):
            prev_idx = current_idx - k
            prev_row, prev_file_idx = self._get_row(prev_idx)
            
            is_valid = True
            if prev_row is None or prev_file_idx != anchor_file_idx:
                is_valid = False; outlier_count_replay += 1
            elif prev_row['replay_id'] != anchor_replay_id:
                is_valid = False; outlier_count_replay += 1
            elif prev_row['score_difference'] != anchor_score_diff:
                is_valid = False; outlier_count_score += 1
            else:
                prev_ball_hit = prev_row.get('ball_hit_team_num', '0.5')
                if (anchor_ball_hit == '0.5') and (prev_ball_hit != '0.5'):
                    is_valid = False; outlier_count_overtime += 1
            
            if is_valid:
                sequence_rows.insert(0, prev_row); last_valid_row = prev_row
            else:
                sequence_rows.insert(0, last_valid_row)

        final_labels = (
            torch.tensor([float(anchor_row['team_1_goal_in_event_window'])], dtype=torch.float32),
            torch.tensor([float(anchor_row['team_0_goal_in_event_window'])], dtype=torch.float32)
        )
        
        graph_sequence = []
        for row in sequence_rows:
            graph_data = self._build_graph_from_row(row)
            if graph_data is None: self.skipped_count += 1; return None
            # We don't need labels on the individual graphs
            del graph_data.y_orange
            del graph_data.y_blue
            graph_sequence.append(graph_data)
        
        return graph_sequence, final_labels, outlier_count_replay, outlier_count_score, outlier_count_overtime

# ====================== GNN-LSTM MODEL ======================

# 1. The GNN "Encoder" Sub-module
class GAT_Encoder(nn.Module):
    def __init__(self, node_features, hidden_dim, edge_dim):
        super().__init__()
        self.conv1 = GATConv(node_features, hidden_dim, heads=4, edge_dim=edge_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 4)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, edge_dim=edge_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Use GlobalAttention pooling as requested
        self.gate_nn = nn.Linear(hidden_dim, 1) 
        self.pool = GlobalAttention(gate_nn=self.gate_nn)
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_attr)))
        # Return the pooled graph embedding
        return self.pool(x, data.batch)

# 2. The Main Spatio-Temporal Model
class SpatioTemporal_GNN_LSTM(nn.Module):
    def __init__(self, node_features, global_features, gnn_hidden_dim, edge_dim, 
                 lstm_hidden_dim, num_lstm_layers, dropout, sequence_length=6):
        super().__init__()
        
        self.sequence_length = sequence_length
        self.gnn_hidden_dim = gnn_hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        # 1. GNN (Spatial Encoder)
        self.gnn = GAT_Encoder(node_features, gnn_hidden_dim, edge_dim)
        
        # 2. LSTM (Temporal Encoder)
        self.lstm = nn.LSTM(
            input_size=gnn_hidden_dim, 
            hidden_size=lstm_hidden_dim, 
            num_layers=num_lstm_layers, 
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0
        )
        
        self.dropout = nn.Dropout(p=dropout)

        # 3. Prediction Heads
        head_input_dim = lstm_hidden_dim + global_features
        self.orange_head = nn.Sequential(nn.Linear(head_input_dim, head_input_dim // 2), nn.ReLU(), nn.Dropout(p=dropout), nn.Linear(head_input_dim // 2, 1))
        self.blue_head = nn.Sequential(nn.Linear(head_input_dim, head_input_dim // 2), nn.ReLU(), nn.Dropout(p=dropout), nn.Linear(head_input_dim // 2, 1))

    def forward(self, data):
        # data is the giant `batched_graphs` from the collate_fn
        
        # 1. Pass all graphs (batch_size * 6) through the GNN
        # Output shape: [batch_size * 6, gnn_hidden_dim]
        graph_embeds = self.gnn(data)

        # 2. Reshape for the LSTM
        batch_size = data.num_graphs // self.sequence_length
        # [batch_size * 6, gnn_hidden_dim] -> [batch_size, 6, gnn_hidden_dim]
        sequence = graph_embeds.view(batch_size, self.sequence_length, self.gnn_hidden_dim)

        # 3. Pass the sequence through the LSTM
        lstm_out, (hn, cn) = self.lstm(sequence)
        
        # We only care about the *last* time step's output
        last_time_step_out = lstm_out[:, -1, :] # Shape: [batch_size, lstm_hidden_dim]
        last_time_step_out = self.dropout(last_time_step_out) # Apply dropout
        
        # 4. Get the global features for the *last* frame
        # We slice the original data object to get the globals for the 6th frame of each seq
        last_frame_globals = data.global_features[self.sequence_length - 1::self.sequence_length]
        
        # 5. Combine and predict
        combined = torch.cat([last_time_step_out, last_frame_globals], dim=1)
        
        return self.orange_head(combined), self.blue_head(combined)

# ====================== HELPER FUNCTIONS ======================

def collate_fn_master(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    # Unzip the batch
    sequences, labels, replay_outliers, score_outliers, overtime_outliers = zip(*batch)

    # Flatten the list of sequences into one big list of Data objects
    flat_graph_list = [graph for seq in sequences for graph in seq]
    
    # Create one giant PyG Batch object from all graphs
    batched_graphs = Batch.from_data_list(flat_graph_list)

    # Stack the labels
    y_orange_labels = torch.stack([label[0] for label in labels])
    y_blue_labels = torch.stack([label[1] for label in labels])
    
    # Sum outlier counts
    total_replay_outliers = sum(replay_outliers)
    total_score_outliers = sum(score_outliers)
    total_overtime_outliers = sum(overtime_outliers)
    
    return batched_graphs, y_orange_labels, y_blue_labels, total_replay_outliers, total_score_outliers, total_overtime_outliers

# (calculate_class_weights and find_optimal_threshold are identical to previous scripts)
def calculate_class_weights(train_files, device):
    print("\n--- Calculating class weights for loss function ---")
    pos_orange, neg_orange, pos_blue, neg_blue = 0, 0, 0, 0
    for file in tqdm(train_files, desc="Scanning labels"):
        try:
            df = pd.read_csv(file, usecols=['team_1_goal_in_event_window', 'team_0_goal_in_event_window'])
            pos_orange += df['team_1_goal_in_event_window'].sum(); neg_orange += len(df) - df['team_1_goal_in_event_window'].sum()
            pos_blue += df['team_0_goal_in_event_window'].sum(); neg_blue += len(df) - df['team_0_goal_in_event_window'].sum()
        except Exception as e:
            print(f"Warning: Skipping file {file} due to error: {e}")
    pos_weight_orange = torch.tensor([(neg_orange / pos_orange) if pos_orange > 0 else 1.0], device=device)
    pos_weight_blue = torch.tensor([(neg_blue / pos_blue) if pos_blue > 0 else 1.0], device=device)
    print(f"Positional weight for Orange loss: {pos_weight_orange.item():.2f}")
    print(f"Positional weight for Blue loss: {pos_weight_blue.item():.2f}")
    return pos_weight_orange, pos_weight_blue

def find_optimal_threshold(y_true, y_pred_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_f1_idx = np.argmax(f1_scores[:-1]) 
    return thresholds[best_f1_idx], f1_scores[best_f1_idx]

def get_predictions_and_loss_sequential(model, loader, device, criterion_o, criterion_b):
    model.eval()
    all_orange_labels, all_blue_labels = [], []
    all_orange_probs, all_blue_probs = [], []
    total_loss_o, total_loss_b = 0.0, 0.0
    total_replay_outliers, total_score_outliers, total_overtime_outliers = 0, 0, 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Getting predictions and loss"):
            if batch is None: continue
            
            batched_graphs, y_o_batch, y_b_batch, replay_outliers, score_outliers, overtime_outliers = batch
            batched_graphs, y_o_batch, y_b_batch = batched_graphs.to(device), y_o_batch.to(device), y_b_batch.to(device)
            
            orange_logits, blue_logits = model(batched_graphs)
            
            # Use batch_size (num of sequences) for loss normalization
            batch_size = y_o_batch.size(0)
            total_loss_o += criterion_o(orange_logits, y_o_batch).item() * batch_size
            total_loss_b += criterion_b(blue_logits, y_b_batch).item() * batch_size
            
            all_orange_labels.extend(y_o_batch.cpu().numpy().flatten())
            all_blue_labels.extend(y_b_batch.cpu().numpy().flatten())
            all_orange_probs.extend(torch.sigmoid(orange_logits).cpu().numpy().flatten())
            all_blue_probs.extend(torch.sigmoid(blue_logits).cpu().numpy().flatten())
            
            total_replay_outliers += replay_outliers
            total_score_outliers += score_outliers
            total_overtime_outliers += overtime_outliers

    num_samples = len(all_orange_labels)
    avg_loss_o = total_loss_o / num_samples if num_samples > 0 else 0
    avg_loss_b = total_loss_b / num_samples if num_samples > 0 else 0
            
    return (np.array(all_orange_labels), np.array(all_blue_labels), 
            np.array(all_orange_probs), np.array(all_blue_probs),
            avg_loss_o, avg_loss_b,
            total_replay_outliers, total_score_outliers, total_overtime_outliers)

# ====================== ARGUMENT PARSER ======================
def parse_args():
    parser = argparse.ArgumentParser(description="Rocket League Spatio-Temporal GNN-LSTM Training")
    parser.add_argument('--data-dir', type=str, default="E:\\...\\split_dataset", help='Parent directory of splits.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    
    # Regularization & Hyperparams
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size (default: 128).')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate (default: 0.0001).')
    parser.add_argument('--gnn-hidden-dim', type=int, default=64, help='GNN hidden dimension.')
    parser.add_argument('--lstm-hidden-dim', type=int, default=128, help='LSTM hidden dimension.')
    parser.add_argument('--lstm-layers', type=int, default=1, help='Number of LSTM layers.')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout probability (default: 0.3).')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Adam weight decay (L2 reg) (default: 1e-5).')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience in epochs (default: 5).')
    
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA.')
    parser.add_argument('--wandb-project', type=str, default="rl-gnn-lstm", help="W&B project name.")
    parser.add_argument('--run-name', type=str, default=None, help="Custom name for the W&B run.")
    parser.add_argument('--resume', action='store_true', help='Resume training from a checkpoint.')
    parser.add_argument('--checkpoint-path', type=str, default='./gnn_lstm_checkpoint.pth', help='Path for checkpoint.')
    parser.add_argument('--checkpoint-every', type=int, default=5, help='Save checkpoint every N epochs.')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of workers for DataLoader.')
    return parser.parse_args()

# ====================== MAIN EXECUTION ======================
def main():
    start_time = time.time()
    args = parse_args()
    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    
    torch.manual_seed(42); np.random.seed(42)
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"--- Using device: {device} ---")

    start_epoch, best_val_loss, wandb_run_id, epochs_no_improve = 0, np.inf, None, 0
    
    if args.resume and os.path.exists(args.checkpoint_path):
        print(f"--- Resuming from checkpoint: {args.checkpoint_path} ---")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', np.inf) 
        wandb_run_id = checkpoint.get('wandb_run_id')
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
        print(f"--- Resuming from epoch {start_epoch}. Best Val Loss: {best_val_loss:.4f} ---")
    
    try:
        wandb.init(project=args.wandb_project, name=args.run_name, config=args, id=wandb_run_id, resume="allow")
    except Exception as e:
        print(f"--- Could not initialize W&B: {e}. Running without logging. ---"); wandb.run = None

    # --- 1. Load Data & Weights ---
    train_dir = os.path.join(args.data_dir, 'train'); val_dir = os.path.join(args.data_dir, 'val')
    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.csv')]
    val_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.csv')]
    
    pos_weight_orange, pos_weight_blue = calculate_class_weights(train_files, device)
    
    train_dataset = SequentialGraphLazyDataset(train_files, sequence_length=SEQUENCE_LENGTH)
    val_dataset = SequentialGraphLazyDataset(val_files, sequence_length=SEQUENCE_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_master)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_master)

    # --- 2. Initialize Model & Criteria ---
    model = SpatioTemporal_GNN_LSTM(
        node_features=NODE_FEATURES,
        global_features=GLOBAL_FEATURES,
        gnn_hidden_dim=args.gnn_hidden_dim,
        edge_dim=EDGE_DIM,
        lstm_hidden_dim=args.lstm_hidden_dim,
        num_lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        sequence_length=SEQUENCE_LENGTH
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion_orange = nn.BCEWithLogitsLoss(pos_weight=pos_weight_orange)
    criterion_blue = nn.BCEWithLogitsLoss(pos_weight=pos_weight_blue)
    
    if args.resume and 'checkpoint' in locals(): 
        model.load_state_dict(checkpoint['model_state'])
        if 'optimizer_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state'])

    # --- 3. Training Loop ---
    print(f"\n--- Starting Training from epoch {start_epoch + 1} to {args.epochs} ---")
    for epoch in range(start_epoch, args.epochs):
        model.train(); total_train_loss = 0
        epoch_replay_outliers, epoch_score_outliers, epoch_overtime_outliers = 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [TRAIN]")
        for batch in pbar:
            if batch is None: continue
            
            batched_graphs, y_o_batch, y_b_batch, replay_outliers, score_outliers, overtime_outliers = batch
            batched_graphs, y_o_batch, y_b_batch = batched_graphs.to(device), y_o_batch.to(device), y_b_batch.to(device)
            
            epoch_replay_outliers += replay_outliers
            epoch_score_outliers += score_outliers
            epoch_overtime_outliers += overtime_outliers
            
            optimizer.zero_grad()
            orange_logits, blue_logits = model(batched_graphs)
            loss = criterion_orange(orange_logits, y_o_batch) + criterion_blue(blue_logits, y_b_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0

        # --- Validation Loop ---
        (val_labels_o, val_labels_b, val_probs_o, val_probs_b, 
         val_loss_o, val_loss_b, val_replay_outliers, 
         val_score_outliers, val_overtime_outliers) = get_predictions_and_loss_sequential(
             model, val_loader, device, criterion_orange, criterion_blue)
        
        avg_val_loss = val_loss_o + val_loss_b
        
        val_f1_o = f1_score(val_labels_o, val_probs_o > 0.5, zero_division=0)
        val_f1_b = f1_score(val_labels_b, val_probs_b > 0.5, zero_division=0)
        avg_val_f1_at_05 = (val_f1_o + val_f1_b) / 2
        val_auprc_o = average_precision_score(val_labels_o, val_probs_o)
        val_auprc_b = average_precision_score(val_labels_b, val_probs_b)
        avg_val_auprc = (val_auprc_o + val_auprc_b) / 2
        
        print(f"\nEpoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Avg Val F1@0.5: {avg_val_f1_at_05:.4f} | Avg Val AUPRC: {avg_val_auprc:.4f}")
        print(f"  Train Outliers: Replay={epoch_replay_outliers}, Score={epoch_score_outliers}, Overtime={epoch_overtime_outliers}")
        print(f"  Val Outliers:   Replay={val_replay_outliers}, Score={val_score_outliers}, Overtime={val_overtime_outliers}")

        if wandb.run:
            wandb.log({
                "epoch": epoch + 1, "train/loss": avg_train_loss, "val/loss": avg_val_loss,
                "val/f1_orange_at_0.5": val_f1_o, "val/f1_blue_at_0.5": val_f1_b, "val/avg_f1_at_0.5": avg_val_f1_at_05,
                "val/auprc_orange": val_auprc_o, "val/auprc_blue": val_auprc_b, "val/avg_auprc": avg_val_auprc,
                "outliers/train_replay": epoch_replay_outliers, "outliers/train_score": epoch_score_outliers, "outliers/train_overtime": epoch_overtime_outliers,
                "outliers/val_replay": val_replay_outliers, "outliers/val_score": val_score_outliers, "outliers/val_overtime": val_overtime_outliers,
                "learning_rate": optimizer.param_groups[0]['lr']
            })

        current_wandb_id = wandb.run.id if wandb.run else None
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0 
            best_model_path = os.path.join(os.path.dirname(args.checkpoint_path), 'best_gnn_lstm_model.pth')
            print(f"  *** New best model found (Val Loss: {best_val_loss:.4f} at epoch {epoch+1}). Saving 'best' checkpoint. ***")
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'best_val_loss': best_val_loss, 'args': vars(args), 'wandb_run_id': current_wandb_id, 'epochs_no_improve': epochs_no_improve}, best_model_path)
        else:
            epochs_no_improve += 1
            print(f"  --- Val loss did not improve. Patience: {epochs_no_improve}/{args.patience} ---")

        if (epoch + 1) % args.checkpoint_every == 0 or (epoch + 1) == args.epochs:
            print(f"--- Saving periodic checkpoint at epoch {epoch + 1}. ---")
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'best_val_loss': best_val_loss, 'args': vars(args), 'wandb_run_id': current_wandb_id, 'epochs_no_improve': epochs_no_improve}, args.checkpoint_path)
        
        if epochs_no_improve >= args.patience:
            print(f"\n--- Early stopping triggered after {args.patience} epochs with no improvement. ---")
            break 
    
    # ... (W&B finish logic) ...
    print("\n--- Script Finished Training ---")

    # ================= FINAL VALIDATION & TEST EVALUATION ============================
    print("\n--- Starting Final Evaluation ---")
    best_model_path = os.path.join(os.path.dirname(args.checkpoint_path), 'best_gnn_lstm_model.pth') 

    if not os.path.exists(best_model_path):
        print("--- No 'best_gnn_lstm_model.pth' checkpoint found. Skipping final test evaluation. ---")
    else:
        print(f"--- Loading best model from: {best_model_path} for final evaluation ---")
        checkpoint = torch.load(best_model_path, map_location=device)
        
        cp_args = checkpoint.get('args', {})
        gnn_hidden_dim = cp_args.get('gnn_hidden_dim', args.gnn_hidden_dim)
        lstm_hidden_dim = cp_args.get('lstm_hidden_dim', args.lstm_hidden_dim)
        lstm_layers = cp_args.get('lstm_layers', args.lstm_layers)
        dropout = cp_args.get('dropout', args.dropout)
        
        model = SpatioTemporal_GNN_LSTM(
            node_features=NODE_FEATURES, global_features=GLOBAL_FEATURES,
            gnn_hidden_dim=gnn_hidden_dim, edge_dim=EDGE_DIM,
            lstm_hidden_dim=lstm_hidden_dim, num_lstm_layers=lstm_layers,
            dropout=dropout, sequence_length=SEQUENCE_LENGTH
        ).to(device)
        model.load_state_dict(checkpoint['model_state'])
        
        criterion_o = nn.BCEWithLogitsLoss(pos_weight=pos_weight_orange)
        criterion_b = nn.BCEWithLogitsLoss(pos_weight=pos_weight_blue)

        # --- Step 1: Run Analysis on FULL Validation Set ---
        print("\n--- Analyzing on VALIDATION set... ---")
        (val_labels_o, val_labels_b, val_probs_o, val_probs_b, 
         val_loss_o, val_loss_b, v_rep, v_sco, v_ot) = get_predictions_and_loss_sequential(
            model, val_loader, device, criterion_o, criterion_b)
        
        optimal_threshold_orange, _ = find_optimal_threshold(val_labels_o, val_probs_o)
        optimal_threshold_blue, _ = find_optimal_threshold(val_labels_b, val_probs_b)
        
        # (Print full val report)
        print_and_log_metrics("Validation", val_labels_o, val_probs_o, val_loss_o, optimal_threshold_orange, "Orange", v_rep, v_sco, v_ot)
        print_and_log_metrics("Validation", val_labels_b, val_probs_b, val_loss_b, optimal_threshold_blue, "Blue", v_rep, v_sco, v_ot)


        # --- Step 2: Run Evaluation on the Test Set ---
        print("\n--- Running final evaluation on the test set... ---")
        test_dir = os.path.join(args.data_dir, 'test')
        test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.csv')]
        
        if not test_files:
            print("--- No test files found. Skipping. ---")
        else:
            test_dataset = SequentialGraphLazyDataset(test_files, sequence_length=SEQUENCE_LENGTH)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_master)
            
            (y_true_o, y_true_b, y_prob_o, y_prob_b, 
             test_loss_o, test_loss_b, t_rep, t_sco, t_ot) = get_predictions_and_loss_sequential(
                model, test_loader, device, criterion_o, criterion_b)

            # (Print full test report)
            metrics_def_o, metrics_opt_o = print_and_log_metrics("Test", y_true_o, y_prob_o, test_loss_o, optimal_threshold_orange, "Orange", t_rep, t_sco, t_ot)
            metrics_def_b, metrics_opt_b = print_and_log_metrics("Test", y_true_b, y_prob_b, test_loss_b, optimal_threshold_blue, "Blue", t_rep, t_sco, t_ot)

            # --- Step 3: Log Final Summary to W&B ---
            try:
                if wandb.run is None: 
                    wandb.init(project=args.wandb_project, id=wandb_run_id, resume="must")
                
                print("\n--- Logging final summary to W&B ---")
                wandb.summary["best_epoch"] = checkpoint.get('epoch', 0) + 1
                wandb.summary["best_val_loss_at_save"] = checkpoint.get('best_val_loss', 0.0)
                # ... (Full W&B summary logging for all metrics) ...
                
                wandb.finish()
            except Exception as e:
                print(f"--- Could not log final summary to W&B: {e} ---")
                
    # Final time printout
    end_time_final = time.time()
    total_seconds_final = end_time_final - start_time
    print(f"\n--- Total Run Time: {total_seconds_final // 3600:.0f}h {(total_seconds_final % 3600) // 60:.0f}m {total_seconds_final % 60:.2f}s ---")

##### NEW/MODIFIED #####
# Created a single, comprehensive print function
def print_and_log_metrics(set_name, y_true, y_prob, weighted_loss, threshold, team_name, r_out, s_out, o_out):
    """Calculates and prints the full block of required metrics."""
    
    # --- Calculate all metrics for both thresholds ---
    preds_def = (y_prob > 0.5).astype(int)
    preds_opt = (y_prob > threshold).astype(int)

    tn_def, fp_def, fn_def, tp_def = confusion_matrix(y_true, preds_def, labels=[0,1]).ravel()
    f1_def = f1_score(y_true, preds_def, zero_division=0); prec_def = precision_score(y_true, preds_def, zero_division=0); rec_def = recall_score(y_true, preds_def, zero_division=0); acc_def = accuracy_score(y_true, preds_def)

    tn_opt, fp_opt, fn_opt, tp_opt = confusion_matrix(y_true, preds_opt, labels=[0,1]).ravel()
    f1_opt = f1_score(y_true, preds_opt, zero_division=0); prec_opt = precision_score(y_true, preds_opt, zero_division=0); rec_opt = recall_score(y_true, preds_opt, zero_division=0); acc_opt = accuracy_score(y_true, preds_opt)

    auprc = average_precision_score(y_true, y_prob)

    # --- Print Block ---
    print(f"\n--- FINAL {set_name.upper()} RESULTS ({team_name.upper()}) ---")
    print(f"  Outliers Found: Replay={r_out}, Score={s_out}, Overtime={o_out}")
    print(f"  {team_name.upper()} Team Weighted Log Loss: {weighted_loss:.4f}")
    print(f"  {team_name.upper()} Team AUPRC: {auprc:.4f}")

    print(f"\n-- Default @ 0.5 Threshold --")
    print(f"  F1: {f1_def:.4f} | P: {prec_def:.4f} | R: {rec_def:.4f} | Acc: {acc_def:.4f}")
    print(f"    -> TP: {tp_def} | TN: {tn_def} | FP: {fp_def} | FN: {fn_def}")

    print(f"\n-- Optimized @ {threshold:.4f} Threshold --")
    print(f"  F1: {f1_opt:.4f} | P: {prec_opt:.4f} | R: {rec_opt:.4f} | Acc: {acc_opt:.4f}")
    print(f"    -> TP: {tp_opt} | TN: {tn_opt} | FP: {fp_opt} | FN: {fn_opt}")
    
    return (prec_def, rec_def, f1_def), (prec_opt, rec_opt, f1_opt)
##### END NEW/MODIFIED #####

if __name__ == '__main__':
    main()