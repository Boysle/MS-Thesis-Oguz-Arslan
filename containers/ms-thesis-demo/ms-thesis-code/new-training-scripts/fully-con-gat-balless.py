import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import linecache
import time  

from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, confusion_matrix, average_precision_score, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool, GlobalAttention
import wandb

# ====================== CONFIGURATION & CONSTANTS ======================
NUM_PLAYERS = 6; PLAYER_FEATURES = 13; GLOBAL_FEATURES = 14
POS_MIN_X, POS_MAX_X = -4096, 4096; POS_MIN_Y, POS_MAX_Y = -6000, 6000; POS_MIN_Z, POS_MAX_Z = 0, 2044
VEL_MIN, VEL_MAX = -2300, 2300; BOOST_MIN, BOOST_MAX = 0, 100; BALL_VEL_MIN, BALL_VEL_MAX = -6000, 6000
BOOST_PAD_MIN, BOOST_PAD_MAX = 0, 10; DIST_MIN, DIST_MAX = 0, (8192**2 + 10240**2 + 2044**2)**0.5

def normalize(val, min_val, max_val):
    return (val - min_val) / (max_val - min_val + 1e-8)

# ====================== ARGUMENT PARSER ======================
def parse_args():
    parser = argparse.ArgumentParser(description="Rocket League GAT Training (Professional)")
    parser.add_argument('--data-dir', type=str, default=r'F:\Raw RL Esports Replays\Day 3 Swiss Stage\Round 1\split_dataset',help='Parent directory of train/val/test splits.')
    
    ##### NEW/FIXED #####
    parser.add_argument('--edge-features', type=int, default=4, choices=[1, 2, 3, 4], help='Number of edge features to use (1-4 for ablation).')
    
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs.') # Default was 5, set to 25 from table
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size.') # Default was 512, set to 1024 from table
    parser.add_argument('--learning-rate', type=float, default=0.0005, help='Learning rate.')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension size for GAT layers.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA.')
    parser.add_argument('--wandb-project', type=str, default="rl-gat-test", help="W&B project name.")
    parser.add_argument('--run-name', type=str, default=None, help="Custom name for the W&B run.")
    parser.add_argument('--resume', action='store_true', help='Resume training from a checkpoint.')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints_gnn/gnn_checkpoint.pth', help='Path for periodic checkpoint.')
    parser.add_argument('--checkpoint-every', type=int, default=5, help='Save checkpoint every N epochs.')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of workers for DataLoader (Set to 0 for Windows).')
    return parser.parse_args()

# ====================== DATASET CLASS (LAZY LOADING FOR GRAPHS) ======================
class GraphLazyDataset(torch.utils.data.Dataset):
    ##### NEW/FIXED #####: Added edge_feature_dim
    def __init__(self, list_of_csv_paths, edge_feature_dim=4):
        self.csv_paths = list_of_csv_paths
        
        ##### NEW/FIXED #####
        self.edge_feature_dim = edge_feature_dim 
        
        self.file_info, self.cumulative_rows, self.header, total_rows = [], [0], None, 0
        if not list_of_csv_paths:
            self.length = 0
            return

        desc = f"Indexing {os.path.basename(os.path.dirname(list_of_csv_paths[0]))} files"
        for path in tqdm(self.csv_paths, desc=desc):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    if self.header is None:
                        self.header = f.readline().strip().split(',')
                    num_lines = sum(1 for _ in f)
                if num_lines > 0:
                    self.file_info.append({'path': path, 'rows': num_lines})
                    total_rows += num_lines
                    self.cumulative_rows.append(total_rows)
            except Exception as e:
                print(f"\nWarning: Could not process file {path}. Skipping. Error: {e}")

        self.length = total_rows
        self.edge_index = torch.tensor([(i, j) for i in range(NUM_PLAYERS) for j in range(NUM_PLAYERS) if i != j], dtype=torch.long).t().contiguous()
        self.skipped_count = 0
        print(f"\n--- Indexing complete. Total samples: {self.length}. Edge features: {self.edge_feature_dim} ---")


    def normalized_dot_product(self, v1, v2):
        v1_norm = v1 / (torch.linalg.norm(v1, dim=-1, keepdim=True) + 1e-8)
        v2_norm = v2 / (torch.linalg.norm(v2, dim=-1, keepdim=True) + 1e-8)
        dot = torch.sum(v1_norm * v2_norm, dim=-1)
        return (dot + 1) / 2

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError(f"Index {idx} out of range for dataset of length {self.length}")

        file_index = np.searchsorted(self.cumulative_rows, idx, side='right') - 1
        file_path = self.file_info[file_index]['path']
        local_idx = idx - self.cumulative_rows[file_index]

        line = linecache.getline(file_path, local_idx + 2)
        if not line.strip():
            self.skipped_count += 1
            return None # Return None, will be filtered by collate_fn

        try:
            row = dict(zip(self.header, line.strip().split(',')))

            # Player features
            x_features = []
            for i in range(NUM_PLAYERS):
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
            x_tensor = torch.tensor(x_features, dtype=torch.float32)

            # Global features
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
            global_tensor = torch.tensor(global_features, dtype=torch.float32).unsqueeze(0)

            # Labels
            orange_y = torch.tensor([float(row['team_1_goal_in_event_window'])], dtype=torch.float32)
            blue_y = torch.tensor([float(row['team_0_goal_in_event_window'])], dtype=torch.float32)

            # ================== EDGE FEATURE CALCULATION ==================
            positions = x_tensor[:, 0:3]  # Indices 0, 1, 2
            velocities = x_tensor[:, 3:6] # Indices 3, 4, 5
            forwards = x_tensor[:, 6:9]  # Indices 6, 7, 8
            teams = x_tensor[:, 10]       # Index 10

            edge_attrs = []
            for i, j in self.edge_index.t():
                # Feature 1: Inverse Distance
                dist = torch.linalg.norm(positions[i] - positions[j])
                d0, p = 1500.0, 2.0
                inv_dist = 1.0 / (1.0 + (dist / d0)**p)

                # Feature 2: Team Relationship
                same_team = 1.0 if teams[i] == teams[j] else 0.0
                
                # Feature 3: Velocity Vector Alignment
                vec_i_to_j = positions[j] - positions[i]
                vel_mag = torch.linalg.norm(velocities[i])
                max_vel = 2300.0
                scaled_vel_mag = vel_mag / (max_vel + 1e-8)
                vel_align = self.normalized_dot_product(velocities[i], vec_i_to_j)
                velocity_feature = vel_align * scaled_vel_mag

                # Feature 4: Forward Vector Alignment
                forward_align = self.normalized_dot_product(forwards[i], vec_i_to_j)

                ##### NEW/FIXED #####
                # Create a list of all 4 features
                all_features = [inv_dist, same_team, velocity_feature, forward_align]
                # Slice the list based on the ablation argument
                edge_attrs.append(all_features[:self.edge_feature_dim])

            edge_attr_tensor = torch.tensor(edge_attrs, dtype=torch.float32)

            return Data(x=x_tensor, edge_index=self.edge_index, edge_attr=edge_attr_tensor,
                        global_features=global_tensor, y_orange=orange_y, y_blue=blue_y)


        except (ValueError, KeyError, IndexError) as e:
            # Removed the noisy debug print statements
            self.skipped_count += 1
            return None # Return None, will be filtered by collate_fn

    def empty_data(self):
        """Return an empty Data object with all required keys."""
        x_tensor = torch.zeros((NUM_PLAYERS, PLAYER_FEATURES), dtype=torch.float32)
        global_tensor = torch.zeros((1, GLOBAL_FEATURES), dtype=torch.float32)
        
        ##### NEW/FIXED #####
        # Use the dynamic edge feature dimension
        edge_attr_tensor = torch.zeros((self.edge_index.size(1), self.edge_feature_dim), dtype=torch.float32)

        return Data(x=x_tensor, edge_index=self.edge_index, edge_attr=edge_attr_tensor,
                    global_features=global_tensor,
                    y_orange=torch.tensor([0.0]), y_blue=torch.tensor([0.0]))


def collate_fn_master(batch):
    # Filter out any None values that came from __getitem__
    batch = [item for item in batch if item is not None]
    
    # If the *entire* batch was bad, return None
    if not batch:
        return None
        
    # Otherwise, create a batch from the good items
    return Batch.from_data_list(batch)


# ====================== MODEL ARCHITECTURE ======================
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

# ====================== HELPER FUNCTIONS ======================
def find_optimal_threshold(y_true, y_pred_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_f1_idx = np.argmax(f1_scores[:-1])
    return thresholds[best_f1_idx], f1_scores[best_f1_idx]

# ##### NEW/FIXED #####
# Removed the old, unused `evaluate_and_log_test_set` function.
# We will use the new standardized evaluation block at the end of main().

# ====================== MAIN EXECUTION ======================
def main():
    ##### NEW/FIXED #####
    start_time = time.time()
    
    args = parse_args()
    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"--- Using device: {device} ---")
    print(f"--- Using {args.edge_features} edge features for GAT ablation ---")

    ##### NEW/FIXED #####
    # Changed from best_val_f1 to best_val_loss
    start_epoch, best_val_loss, wandb_run_id = 0, np.inf, None
    
    if args.resume and os.path.exists(args.checkpoint_path):
        print(f"--- Resuming from checkpoint: {args.checkpoint_path} ---")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        
        ##### NEW/FIXED #####
        # Load best_val_loss, fallback to infinity
        best_val_loss = checkpoint.get('best_val_loss', np.inf)
        wandb_run_id = checkpoint.get('wandb_run_id')
        saved_args = checkpoint.get('args', {})
        vars(args).update({k: v for k, v in saved_args.items() if k not in ['resume', 'epochs']})
        print(f"--- Resuming from epoch {start_epoch}. Best Val Loss: {best_val_loss:.4f} ---")
    
    try:
        wandb.init(project=args.wandb_project, name=args.run_name, config=args, id=wandb_run_id, resume="allow")
    except Exception as e:
        print(f"--- Could not initialize W&B: {e}. Running without logging. ---"); wandb.run = None

    train_dir = os.path.join(args.data_dir, 'train'); val_dir = os.path.join(args.data_dir, 'val')
    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.csv')]
    val_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.csv')]
    
    ##### NEW/FIXED #####
    # Pass the edge_feature_dim argument
    train_dataset = GraphLazyDataset(train_files, args.edge_features)
    val_dataset = GraphLazyDataset(val_files, args.edge_features)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_master)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_master)

    print("\n--- Calculating class weights for loss function ---")
    pos_orange, neg_orange, pos_blue, neg_blue = 0, 0, 0, 0
    for file in tqdm(train_files, desc="Scanning labels"):
        df = pd.read_csv(file, usecols=['team_1_goal_in_event_window', 'team_0_goal_in_event_window'])
        pos_orange += df['team_1_goal_in_event_window'].sum(); neg_orange += len(df) - df['team_1_goal_in_event_window'].sum()
        pos_blue += df['team_0_goal_in_event_window'].sum(); neg_blue += len(df) - df['team_0_goal_in_event_window'].sum()
    
    pos_weight_orange = torch.tensor([neg_orange / pos_orange], device=device) if pos_orange > 0 else torch.tensor([1.0], device=device)
    pos_weight_blue = torch.tensor([neg_blue / pos_blue], device=device) if pos_blue > 0 else torch.tensor([1.0], device=device)
    print(f"Positional weight for Orange loss: {pos_weight_orange.item():.2f}")
    print(f"Positional weight for Blue loss: {pos_weight_blue.item():.2f}")

    ##### NEW/FIXED #####
    # Pass the dynamic edge_dim
    model = RocketLeagueGAT(PLAYER_FEATURES, GLOBAL_FEATURES, args.hidden_dim, edge_dim=args.edge_features).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion_orange = nn.BCEWithLogitsLoss(pos_weight=pos_weight_orange)
    criterion_blue = nn.BCEWithLogitsLoss(pos_weight=pos_weight_blue)
    
    if args.resume and 'checkpoint' in locals():
        model.load_state_dict(checkpoint['model_state'])
        if 'optimizer_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state'])

    print(f"\n--- Starting Training from epoch {start_epoch + 1} to {args.epochs} ---")
    for epoch in range(start_epoch, args.epochs):
        model.train(); total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [TRAIN]")
        for batch in pbar:
            if batch is None or not hasattr(batch, 'x'): continue
            batch = batch.to(device)
            optimizer.zero_grad()
            orange_logits, blue_logits = model(batch)
            loss = criterion_orange(orange_logits, batch.y_orange.view_as(orange_logits)) + criterion_blue(blue_logits, batch.y_blue.view_as(blue_logits))
            loss.backward(); optimizer.step(); total_train_loss += loss.item()
        
        model.eval(); total_val_loss = 0
        all_val_oprobs, all_val_olabels, all_val_bprobs, all_val_blabels = [], [], [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [VAL]"):
                if batch is None: continue
                batch = batch.to(device)
                orange_logits, blue_logits = model(batch)
                total_val_loss += (criterion_orange(orange_logits, batch.y_orange.view_as(orange_logits)) + criterion_blue(blue_logits, batch.y_blue.view_as(blue_logits))).item()
                all_val_oprobs.extend(torch.sigmoid(orange_logits).cpu().numpy().flatten()); all_val_olabels.extend(batch.y_orange.cpu().numpy().flatten())
                all_val_bprobs.extend(torch.sigmoid(blue_logits).cpu().numpy().flatten()); all_val_blabels.extend(batch.y_blue.cpu().numpy().flatten())
        
        avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0

        ##### NEW/FIXED #####
        # --- Calculate all validation metrics ---
        np_val_olabels = np.array(all_val_olabels)
        np_val_oprobs = np.array(all_val_oprobs)
        np_val_blabels = np.array(all_val_blabels)
        np_val_bprobs = np.array(all_val_bprobs)

        # Calculate @ 0.5 threshold
        np_val_opreds_binary = (np_val_oprobs > 0.5).astype(int)
        np_val_bpreds_binary = (np_val_bprobs > 0.5).astype(int)
        
        val_f1_o = f1_score(np_val_olabels, np_val_opreds_binary, zero_division=0)
        val_f1_b = f1_score(np_val_blabels, np_val_bpreds_binary, zero_division=0)
        avg_val_f1_at_05 = (val_f1_o + val_f1_b) / 2
        
        val_prec_o = precision_score(np_val_olabels, np_val_opreds_binary, zero_division=0)
        val_recall_o = recall_score(np_val_olabels, np_val_opreds_binary, zero_division=0)
        val_prec_b = precision_score(np_val_blabels, np_val_bpreds_binary, zero_division=0)
        val_recall_b = recall_score(np_val_blabels, np_val_bpreds_binary, zero_division=0)

        # Calculate AUPRC (threshold-independent)
        val_auprc_o = average_precision_score(np_val_olabels, np_val_oprobs)
        val_auprc_b = average_precision_score(np_val_blabels, np_val_bprobs)
        avg_val_auprc = (val_auprc_o + val_auprc_b) / 2
        
        print(f"\nEpoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Avg Val F1@0.5: {avg_val_f1_at_05:.4f} | Avg Val AUPRC: {avg_val_auprc:.4f}")

        if wandb.run:
            class_names = ["No Goal", "Goal"]
            y_probas_orange_plots = np.stack([1 - np_val_oprobs, np_val_oprobs], axis=1)
            y_probas_blue_plots = np.stack([1 - np_val_bprobs, np_val_bprobs], axis=1)

            wandb.log({
                "epoch": epoch + 1,
                "train/loss": avg_train_loss,
                "val/loss": avg_val_loss,
                "val/f1_orange_at_0.5": val_f1_o,
                "val/f1_blue_at_0.5": val_f1_b,
                "val/avg_f1_at_0.5": avg_val_f1_at_05,
                "val/precision_orange_at_0.5": val_prec_o,
                "val/recall_orange_at_0.5": val_recall_o,
                "val/precision_blue_at_0.5": val_prec_b,
                "val/recall_blue_at_0.5": val_recall_b,
                "val/auprc_orange": val_auprc_o,
                "val/auprc_blue": val_auprc_b,
                "val/avg_auprc": avg_val_auprc,
                # "val/cm_orange": wandb.plot.confusion_matrix(...), # Optional: too heavy
            })

        current_wandb_id = wandb.run.id if wandb.run else None
        
        ##### NEW/FIXED #####
        # This is the main checkpointing fix.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(os.path.dirname(args.checkpoint_path), f'best_gat_model_edge_{args.edge_features}.pth') # Specific name
            print(f"  *** New best model found (Val Loss: {best_val_loss:.4f} at epoch {epoch+1}). Saving 'best' checkpoint. ***")
            torch.save({
                'epoch': epoch, 
                'model_state': model.state_dict(), 
                'best_val_loss': best_val_loss, 
                'args': vars(args), 
                'wandb_run_id': current_wandb_id
            }, best_model_path)
        
        if (epoch + 1) % args.checkpoint_every == 0 or (epoch + 1) == args.epochs:
            print(f"--- Saving periodic checkpoint at epoch {epoch + 1}. ---")
            torch.save({
                'epoch': epoch, 
                'model_state': model.state_dict(), 
                'optimizer_state': optimizer.state_dict(), 
                'best_val_loss': best_val_loss, 
                'args': vars(args), 
                'wandb_run_id': current_wandb_id
            }, args.checkpoint_path)

    print("\n--- Data Loading Summary ---")
    print(f"Total processed training samples: {len(train_dataset)}")
    print(f"Number of skipped/problematic training rows: {train_dataset.skipped_count}")
    print(f"Total processed validation samples: {len(val_dataset)}")
    print(f"Number of skipped/problematic validation rows: {val_dataset.skipped_count}")

    # Finish the training-loop W&B run
    if wandb.run and wandb.run.id == wandb_run_id and wandb.run.resumed is False:
        wandb.finish()
    
    
    ##### NEW/FIXED #####
    # ================= FINAL VALIDATION & TEST EVALUATION ============================
    print("\n--- Training Complete ---")
    best_model_path = os.path.join(os.path.dirname(args.checkpoint_path), f'best_gat_model_edge_{args.edge_features}.pth')

    if not os.path.exists(best_model_path):
        print(f"--- No '{best_model_path}' checkpoint found. Skipping final test evaluation. ---")
    else:
        print(f"--- Loading best model from: {best_model_path} for final evaluation ---")
        checkpoint = torch.load(best_model_path, map_location=device)
        # Re-initialize model to ensure correct architecture
        model = RocketLeagueGAT(PLAYER_FEATURES, GLOBAL_FEATURES, args.hidden_dim, edge_dim=args.edge_features).to(device)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()

        # --- Step 1: Find Optimal Threshold on the FULL Validation Set ---
        print("\n--- Determining optimal thresholds on the validation set... ---")
        all_val_oprobs, all_val_olabels, all_val_bprobs, all_val_blabels = [], [], [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="[FINAL VAL]"):
                if batch is None: continue
                batch = batch.to(device)
                orange_logits, blue_logits = model(batch)
                all_val_oprobs.extend(torch.sigmoid(orange_logits).cpu().numpy().flatten())
                all_val_olabels.extend(batch.y_orange.cpu().numpy().flatten())
                all_val_bprobs.extend(torch.sigmoid(blue_logits).cpu().numpy().flatten())
                all_val_blabels.extend(batch.y_blue.cpu().numpy().flatten())
        
        optimal_threshold_orange, _ = find_optimal_threshold(all_val_olabels, all_val_oprobs)
        optimal_threshold_blue, _ = find_optimal_threshold(all_val_blabels, all_val_bprobs)

        print(f"  Optimal Threshold (Orange): {optimal_threshold_orange:.4f}")
        print(f"  Optimal Threshold (Blue):   {optimal_threshold_blue:.4f}")

        # --- Step 2: Run Evaluation on the Test Set ---
        print("\n--- Running final evaluation on the test set... ---")
        test_dir = os.path.join(args.data_dir, 'test')
        test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.csv')]
        
        if not test_files:
            print("--- No test files found. Skipping. ---")
            test_dataset = None
        else:
            test_dataset = GraphLazyDataset(test_files, args.edge_features)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn_master)
            print(f"Total test samples: {len(test_dataset)}")
            
            all_test_oprobs, all_test_olabels, all_test_bprobs, all_test_blabels = [], [], [], []
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="[FINAL TEST]"):
                    if batch is None: continue
                    batch = batch.to(device)
                    orange_logits, blue_logits = model(batch)
                    all_test_oprobs.extend(torch.sigmoid(orange_logits).cpu().numpy().flatten())
                    all_test_olabels.extend(batch.y_orange.cpu().numpy().flatten())
                    all_test_bprobs.extend(torch.sigmoid(blue_logits).cpu().numpy().flatten())
                    all_test_blabels.extend(batch.y_blue.cpu().numpy().flatten())

            # --- Step 3: Calculate Metrics for Test Set (Optimized & Default) ---
            y_true_o, y_prob_o = np.array(all_test_olabels), np.array(all_test_oprobs)
            y_true_b, y_prob_b = np.array(all_test_blabels), np.array(all_test_bprobs)

            # --- Default @ 0.5 ---
            preds_def_o = (y_prob_o > 0.5).astype(int)
            preds_def_b = (y_prob_b > 0.5).astype(int)
            
            tn_def_o, fp_def_o, fn_def_o, tp_def_o = confusion_matrix(y_true_o, preds_def_o, labels=[0,1]).ravel()
            f1_def_o = f1_score(y_true_o, preds_def_o, zero_division=0)
            prec_def_o = precision_score(y_true_o, preds_def_o, zero_division=0)
            rec_def_o = recall_score(y_true_o, preds_def_o, zero_division=0)
            acc_def_o = accuracy_score(y_true_o, preds_def_o)
            
            tn_def_b, fp_def_b, fn_def_b, tp_def_b = confusion_matrix(y_true_b, preds_def_b, labels=[0,1]).ravel()
            f1_def_b = f1_score(y_true_b, preds_def_b, zero_division=0)
            prec_def_b = precision_score(y_true_b, preds_def_b, zero_division=0)
            rec_def_b = recall_score(y_true_b, preds_def_b, zero_division=0)
            acc_def_b = accuracy_score(y_true_b, preds_def_b)

            # --- Optimized ---
            preds_opt_o = (y_prob_o > optimal_threshold_orange).astype(int)
            preds_opt_b = (y_prob_b > optimal_threshold_blue).astype(int)
            
            tn_opt_o, fp_opt_o, fn_opt_o, tp_opt_o = confusion_matrix(y_true_o, preds_opt_o, labels=[0,1]).ravel()
            f1_opt_o = f1_score(y_true_o, preds_opt_o, zero_division=0)
            prec_opt_o = precision_score(y_true_o, preds_opt_o, zero_division=0)
            rec_opt_o = recall_score(y_true_o, preds_opt_o, zero_division=0)
            acc_opt_o = accuracy_score(y_true_o, preds_opt_o)
            
            tn_opt_b, fp_opt_b, fn_opt_b, tp_opt_b = confusion_matrix(y_true_b, preds_opt_b, labels=[0,1]).ravel()
            f1_opt_b = f1_score(y_true_b, preds_opt_b, zero_division=0)
            prec_opt_b = precision_score(y_true_b, preds_opt_b, zero_division=0)
            rec_opt_b = recall_score(y_true_b, preds_opt_b, zero_division=0)
            acc_opt_b = accuracy_score(y_true_b, preds_opt_b)

            # --- AUPRC (Threshold-Independent) ---
            auprc_o = average_precision_score(y_true_o, y_prob_o)
            auprc_b = average_precision_score(y_true_b, y_prob_b)

            # --- Step 3b: New Print Block ---
            print("\n--- FINAL TEST RESULTS ---")
            print("\n-- Default @ 0.5 Threshold --")
            print(f"  Orange Team: F1: {f1_def_o:.4f} | P: {prec_def_o:.4f} | R: {rec_def_o:.4f} | Acc: {acc_def_o:.4f}")
            print(f"    -> TP: {tp_def_o} | TN: {tn_def_o} | FP: {fp_def_o} | FN: {fn_def_o}")
            print(f"  Blue Team:   F1: {f1_def_b:.4f} | P: {prec_def_b:.4f} | R: {rec_def_b:.4f} | Acc: {acc_def_b:.4f}")
            print(f"    -> TP: {tp_def_b} | TN: {tn_def_b} | FP: {fp_def_b} | FN: {fn_def_b}")

            print("\n-- Optimized Threshold --")
            print(f"  Orange Team (@ {optimal_threshold_orange:.3f}): F1: {f1_opt_o:.4f} | P: {prec_opt_o:.4f} | R: {rec_opt_o:.4f} | Acc: {acc_opt_o:.4f}")
            print(f"    -> TP: {tp_opt_o} | TN: {tn_opt_o} | FP: {fp_opt_o} | FN: {fn_opt_o}")
            print(f"  Blue Team   (@ {optimal_threshold_blue:.3f}): F1: {f1_opt_b:.4f} | P: {prec_opt_b:.4f} | R: {rec_opt_b:.4f} | Acc: {acc_opt_b:.4f}")
            print(f"    -> TP: {tp_opt_b} | TN: {tn_opt_b} | FP: {fp_opt_b} | FN: {fn_opt_b}")

            print("\n-- Threshold-Independent --")
            print(f"  AUPRC (Orange): {auprc_o:.4f}")
            print(f"  AUPRC (Blue):   {auprc_b:.4f}")


            # --- Step 4: Log Final Summary to W&B ---
            try:
                # Re-init wandb to log this summary to the same run
                if wandb.run is None: # Start a new run if the first one finished
                    wandb.init(project=args.wandb_project, id=wandb_run_id, resume="must")
                
                print("\n--- Logging final summary to W&B ---")
                wandb.summary["best_epoch"] = checkpoint.get('epoch', 0) + 1
                wandb.summary["best_val_loss_at_save"] = checkpoint.get('best_val_loss', 0.0)
                
                # Log Default 0.5 scores
                wandb.summary["default_test_f1_orange"] = f1_def_o
                wandb.summary["default_test_precision_orange"] = prec_def_o
                wandb.summary["default_test_recall_orange"] = rec_def_o
                wandb.summary["default_test_accuracy_orange"] = acc_def_o
                wandb.summary["default_test_tp_orange"] = tp_def_o
                wandb.summary["default_test_tn_orange"] = tn_def_o
                
                wandb.summary["default_test_f1_blue"] = f1_def_b
                wandb.summary["default_test_precision_blue"] = prec_def_b
                wandb.summary["default_test_recall_blue"] = rec_def_b
                wandb.summary["default_test_accuracy_blue"] = acc_def_b
                wandb.summary["default_test_tp_blue"] = tp_def_b
                wandb.summary["default_test_tn_blue"] = tn_def_b

                # Log Optimized scores
                wandb.summary["optimal_threshold_orange"] = optimal_threshold_orange
                wandb.summary["optimized_test_f1_orange"] = f1_opt_o
                wandb.summary["optimized_test_precision_orange"] = prec_opt_o
                wandb.summary["optimized_test_recall_orange"] = rec_opt_o
                wandB.summary["optimized_test_accuracy_orange"] = acc_opt_o
                wandb.summary["optimized_test_auprc_orange"] = auprc_o
                wandb.summary["optimized_test_tp_orange"] = tp_opt_o
                wandb.summary["optimized_test_tn_orange"] = tn_opt_o

                wandb.summary["optimal_threshold_blue"] = optimal_threshold_blue
                wandb.summary["optimized_test_f1_blue"] = f1_opt_b
                wandb.summary["optimized_test_precision_blue"] = prec_opt_b
                wandb.summary["optimized_test_recall_blue"] = rec_opt_b
                wandb.summary["optimized_test_accuracy_blue"] = acc_opt_b
                wandb.summary["optimized_test_auprc_blue"] = auprc_b
                wandb.summary["optimized_test_tp_blue"] = tp_opt_b
                wandb.summary["optimized_test_tn_blue"] = tn_opt_b
                
                # Log final plots
                class_names = ["No Goal", "Goal"]
                y_probas_orange_plots = np.stack([1 - y_prob_o, y_prob_o], axis=1)
                y_probas_blue_plots = np.stack([1 - y_prob_b, y_prob_b], axis=1)

                wandb.log({
                    "test/cm_orange_optimized": wandb.plot.confusion_matrix(y_true=y_true_o, preds=preds_opt_o, class_names=class_names),
                    "test/cm_blue_optimized": wandb.plot.confusion_matrix(y_true=y_true_b, preds=preds_opt_b, class_names=class_names),
                    "test/pr_curve_orange": wandb.plot.pr_curve(y_true=y_true_o, y_probas=y_probas_orange_plots, labels=class_names),
                    "test/pr_curve_blue": wandb.plot.pr_curve(y_true=y_true_b, y_probas=y_probas_blue_plots, labels=class_names),
                    "test/roc_curve_orange": wandb.plot.roc_curve(y_true=y_true_o, y_probas=y_probas_orange_plots, labels=class_names),
                    "test/roc_curve_blue": wandb.plot.roc_curve(y_true=y_true_b, y_probas=y_probas_blue_plots, labels=class_names),
                })
                
                # Log time
                end_time = time.time()
                total_seconds = end_time - start_time
                wandb.summary["total_run_time_seconds"] = total_seconds
                
                wandb.finish()
                
            except Exception as e:
                print(f"--- Could not log final summary to W&B: {e} ---")

    # Final time printout
    end_time_final = time.time()
    total_seconds_final = end_time_final - start_time
    print(f"\n--- Total Run Time: {total_seconds_final // 3600:.0f}h {(total_seconds_final % 3600) // 60:.0f}m {total_seconds_final % 60:.2f}s ---")


if __name__ == '__main__':
    main()