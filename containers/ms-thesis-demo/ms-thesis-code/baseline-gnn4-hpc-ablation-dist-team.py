import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import linecache
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv , global_mean_pool
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
    parser = argparse.ArgumentParser(description="Rocket League GCN Training (Professional)")
    parser.add_argument('--data-dir', type=str, default=r'C:\\Users\\serda\\Desktop\\Thesis Dataset Backup\\Raw RL Esports Replays\\Day 3 Swiss Stage\\Round 1\\split_dataset',help='Parent directory of train/val/test splits.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size.')
    parser.add_argument('--learning-rate', type=float, default=0.0005, help='Learning rate.')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension size for GCN layers.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA.')
    parser.add_argument('--wandb-project', type=str, default="rl-goal-prediction-gnn-test", help="W&B project name.")
    parser.add_argument('--run-name', type=str, default=None, help="Custom name for the W&B run.")
    parser.add_argument('--resume', action='store_true', help='Resume training from a checkpoint.')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints_gnn/gnn_checkpoint.pth', help='Path for periodic checkpoint.')
    parser.add_argument('--checkpoint-every', type=int, default=5, help='Save checkpoint every N epochs.')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of workers for DataLoader (Set to 0 for Windows).')
    return parser.parse_args()

# ====================== DATASET CLASS (LAZY LOADING FOR GRAPHS) ======================
class GraphLazyDataset(torch.utils.data.Dataset):
    def __init__(self, list_of_csv_paths):
        self.csv_paths = list_of_csv_paths
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
            # Handle empty lines separately if you want
            print(f"\n--- DEBUG: Encountered an empty line. ---")
            print(f"File Path: {file_path}")
            print(f"Line Number in File: {local_idx + 2}")
            print(f"--- Returning empty data object. ---\n")
            return self.empty_data()

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

            # ================== EDGE FEATURE CALCULATION (DISTANCE + TEAM) ==================

            positions = x_tensor[:, 0:3]  # Assuming pos_x, pos_y, pos_z are the first 3 features
            teams = x_tensor[:, 10]       # Assuming team is the 11th feature (index 10)

            edge_attrs = []
            for i, j in self.edge_index.t():
                # Feature 1: Inverse Distance (Stays the same)
                dist = torch.linalg.norm(positions[i] - positions[j])
                d0, p = 1500.0, 2.0
                inv_dist = 1.0 / (1.0 + (dist / d0)**p)

                # Feature 2: Team Relationship
                same_team = 1.0 if teams[i] == teams[j] else 0.0

                # Append a list with the two features
                edge_attrs.append([inv_dist, same_team])

            # The shape is [num_edges, 2]
            edge_attr_tensor = torch.tensor(edge_attrs, dtype=torch.float32)


            # =======================================================================================

            return Data(x=x_tensor, edge_index=self.edge_index, edge_attr=edge_attr_tensor,
                        global_features=global_tensor, y_orange=orange_y, y_blue=blue_y)


        except (ValueError, KeyError, IndexError):
        # Handle errors, maybe return a graph with zeroed features
        # Note: You may need to create a zeroed edge_attr tensor here as well
            print(f"\n--- DEBUG: Error processing data row. ---")
            print(f"File Path: {file_path}")
            # The line number is local_idx + 2 because:
            # +1 to convert 0-based local_idx to 1-based line number
            # +1 because we skip the header line
            print(f"Line Number in File: {local_idx + 2}") 
            print(f"Original Line Content: '{line.strip()}'")
            print(f"Exception Type: {type(e).__name__}")
            print(f"Exception Details: {e}")
            print(f"--- Returning empty data object to prevent crash. ---\n")
            self.skipped_count += 1
            return self.empty_data() # Make sure empty_data() also includes a zeroed edge_attr

    def empty_data(self):
        """Return an empty Data object with all required keys."""
        x_tensor = torch.zeros((NUM_PLAYERS, PLAYER_FEATURES), dtype=torch.float32)
        global_tensor = torch.zeros((1, GLOBAL_FEATURES), dtype=torch.float32)
        
        # MODIFIED: The second dimension is now 2
        edge_attr_tensor = torch.zeros((self.edge_index.size(1), 2), dtype=torch.float32)

        return Data(x=x_tensor, edge_index=self.edge_index, edge_attr=edge_attr_tensor,
                    global_features=global_tensor,
                    y_orange=torch.tensor([0.0]), y_blue=torch.tensor([0.0]))


def collate_fn_master(batch):
    if not batch:
        return Batch.from_data_list([])
    return Batch.from_data_list(batch)

def normalized_dot_product(v1, v2):
    # Normalize input vectors to handle cases where they are not unit vectors
    v1_norm = v1 / (torch.linalg.norm(v1, dim=-1, keepdim=True) + 1e-8)
    v2_norm = v2 / (torch.linalg.norm(v2, dim=-1, keepdim=True) + 1e-8)
    # Dot product of unit vectors is in [-1, 1]. We scale it to [0, 1].
    dot = torch.sum(v1_norm * v2_norm, dim=-1)
    return (dot + 1) / 2




# ====================== MODEL ARCHITECTURE ======================
class RocketLeagueGAT(nn.Module):
    # Note the new argument `edge_dim`
    def __init__(self, player_features, global_features, hidden_dim, edge_dim):
        super().__init__()
        # We specify the dimensionality of our edge features with `edge_dim`
        # We also add heads=4 to use multi-head attention, a standard practice for GATs
        self.conv1 = GATConv(player_features, hidden_dim, heads=4, edge_dim=edge_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 4) # Output dim is hidden_dim * heads
        
        # The input to the second layer is hidden_dim * heads
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, edge_dim=edge_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # The rest of the model remains the same
        self.orange_head = nn.Sequential(nn.Linear(hidden_dim + global_features, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))
        self.blue_head = nn.Sequential(nn.Linear(hidden_dim + global_features, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))

    def forward(self, data):
        # We now have edge_attr available in the data object
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Pass the edge_attr tensor to the GATConv layers
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_attr)))
        
        # The rest of the forward pass is identical
        graph_embed = global_mean_pool(x, data.batch)
        combined = torch.cat([graph_embed, data.global_features], dim=1)
        return self.orange_head(combined), self.blue_head(combined)

# ====================== HELPER FUNCTIONS ======================
def find_optimal_threshold(y_true, y_pred_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_f1_idx = np.argmax(f1_scores[:-1])
    return thresholds[best_f1_idx], f1_scores[best_f1_idx]

def evaluate_and_log_test_set(model, test_loader, optimal_threshold_orange, optimal_threshold_blue, device, best_checkpoint):
    """
    Runs a full evaluation on the test set, logging all metrics and plots to W&B.
    """
    print("\n--- Running Final Evaluation on TEST Set ---")
    model.eval()
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

    # --- Convert to NumPy arrays ---
    y_true_o = np.array(all_test_olabels); y_true_b = np.array(all_test_blabels)
    y_prob_o = np.array(all_test_oprobs); y_prob_b = np.array(all_test_bprobs)

    # --- Calculate metrics for BOTH thresholds ---
    # Default 0.5
    preds_def_o = (y_prob_o > 0.5).astype(int); preds_def_b = (y_prob_b > 0.5).astype(int)
    f1_def_o = f1_score(y_true_o, preds_def_o, zero_division=0); prec_def_o = precision_score(y_true_o, preds_def_o, zero_division=0); rec_def_o = recall_score(y_true_o, preds_def_o, zero_division=0)
    f1_def_b = f1_score(y_true_b, preds_def_b, zero_division=0); prec_def_b = precision_score(y_true_b, preds_def_b, zero_division=0); rec_def_b = recall_score(y_true_b, preds_def_b, zero_division=0)
    print("\n--- Test Set Results (Default 0.5 Threshold) ---")
    print(f"  Default Orange -> F1: {f1_def_o:.4f} | Precision: {prec_def_o:.4f} | Recall: {rec_def_o:.4f}")
    print(f"  Default Blue   -> F1: {f1_def_b:.4f} | Precision: {prec_def_b:.4f} | Recall: {rec_def_b:.4f}")

    # Optimized
    preds_opt_o = (y_prob_o > optimal_threshold_orange).astype(int); preds_opt_b = (y_prob_b > optimal_threshold_blue).astype(int)
    f1_opt_o = f1_score(y_true_o, preds_opt_o, zero_division=0); prec_opt_o = precision_score(y_true_o, preds_opt_o, zero_division=0); rec_opt_o = recall_score(y_true_o, preds_opt_o, zero_division=0)
    f1_opt_b = f1_score(y_true_b, preds_opt_b, zero_division=0); prec_opt_b = precision_score(y_true_b, preds_opt_b, zero_division=0); rec_opt_b = recall_score(y_true_b, preds_opt_b, zero_division=0)
    print("\n--- Test Set Results (Optimized Thresholds) ---")
    print(f"  Optimized Orange (Thresh={optimal_threshold_orange:.4f}) -> F1: {f1_opt_o:.4f} | Precision: {prec_opt_o:.4f} | Recall: {rec_opt_o:.4f}")
    print(f"  Optimized Blue   (Thresh={optimal_threshold_blue:.4f}) -> F1: {f1_opt_b:.4f} | Precision: {prec_opt_b:.4f} | Recall: {rec_opt_b:.4f}")

    # --- Log everything to W&B ---
    if wandb.run:
        print("\n--- Logging all test results to W&B ---")
        class_names = ["No Goal", "Goal"]
        y_probas_o_plots = np.stack([1 - y_prob_o, y_prob_o], axis=1)
        y_probas_b_plots = np.stack([1 - y_prob_b, y_prob_b], axis=1)

        # Log default and optimized plots
        wandb.log({
            "test/cm_orange_default": wandb.plot.confusion_matrix(y_true=y_true_o, preds=preds_def_o, class_names=class_names),
            "test/cm_blue_default": wandb.plot.confusion_matrix(y_true=y_true_b, preds=preds_def_b, class_names=class_names),
            "test/cm_orange_optimized": wandb.plot.confusion_matrix(y_true=y_true_o, preds=preds_opt_o, class_names=class_names),
            "test/cm_blue_optimized": wandb.plot.confusion_matrix(y_true=y_true_b, preds=preds_opt_b, class_names=class_names),
            "test/pr_curve_orange": wandb.plot.pr_curve(y_true=y_true_o, y_probas=y_probas_o_plots, labels=class_names),
            "test/pr_curve_blue": wandb.plot.pr_curve(y_true=y_true_b, y_probas=y_probas_b_plots, labels=class_names),
            "test/roc_curve_orange": wandb.plot.roc_curve(y_true=y_true_o, y_probas=y_probas_o_plots, labels=class_names),
            "test/roc_curve_blue": wandb.plot.roc_curve(y_true=y_true_b, y_probas=y_probas_b_plots, labels=class_names),
        })

        # Update summary with the most important final scores
        wandb.summary["best_epoch"] = best_checkpoint.get('epoch', 0) + 1
        wandb.summary["best_val_f1_at_save"] = best_checkpoint.get('best_val_f1', 0.0)
        wandb.summary["optimal_threshold_orange"] = optimal_threshold_orange
        wandb.summary["optimized_test_f1_orange"] = f1_opt_o
        wandb.summary["optimized_test_precision_orange"] = prec_opt_o
        wandb.summary["optimized_test_recall_orange"] = rec_opt_o
        wandb.summary["optimal_threshold_blue"] = optimal_threshold_blue
        wandb.summary["optimized_test_f1_blue"] = f1_opt_b
        wandb.summary["optimized_test_precision_blue"] = prec_opt_b
        wandb.summary["optimized_test_recall_blue"] = rec_opt_b

# ====================== MAIN EXECUTION ======================
def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"--- Using device: {device} ---")

    start_epoch, best_val_f1, wandb_run_id = 0, 0.0, None
    if args.resume and os.path.exists(args.checkpoint_path):
        print(f"--- Resuming from checkpoint: {args.checkpoint_path} ---")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        start_epoch = checkpoint['epoch'] + 1; best_val_f1 = checkpoint.get('best_val_f1', 0.0); wandb_run_id = checkpoint.get('wandb_run_id')
        saved_args = checkpoint.get('args', {})
        vars(args).update({k: v for k, v in saved_args.items() if k not in ['resume', 'epochs']})
        print(f"--- Resuming from epoch {start_epoch}. Best F1: {best_val_f1:.4f} ---")
    
    try:
        wandb.init(project=args.wandb_project, name=args.run_name, config=args, id=wandb_run_id, resume="allow")
    except Exception as e:
        print(f"--- Could not initialize W&B: {e}. Running without logging. ---"); wandb.run = None

    train_dir = os.path.join(args.data_dir, 'train'); val_dir = os.path.join(args.data_dir, 'val')
    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.csv')]
    val_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.csv')]
    train_dataset = GraphLazyDataset(train_files); val_dataset = GraphLazyDataset(val_files)
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

    model = RocketLeagueGAT(PLAYER_FEATURES, GLOBAL_FEATURES, args.hidden_dim, edge_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion_orange = nn.BCEWithLogitsLoss(pos_weight=pos_weight_orange)
    criterion_blue = nn.BCEWithLogitsLoss(pos_weight=pos_weight_blue)
    
    if args.resume and 'checkpoint' in locals():
        model.load_state_dict(checkpoint['model_state']); optimizer.load_state_dict(checkpoint['optimizer_state'])

    print(f"\n--- Starting Training from epoch {start_epoch + 1} to {args.epochs} ---")
    for epoch in range(start_epoch, args.epochs):
        model.train(); total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [TRAIN]")
        for batch in pbar:
            if batch is None: continue
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
        val_f1_o = f1_score(all_val_olabels, np.array(all_val_oprobs) > 0.5, zero_division=0); val_f1_b = f1_score(all_val_blabels, np.array(all_val_bprobs) > 0.5, zero_division=0)
        avg_val_f1 = (val_f1_o + val_f1_b) / 2
        print(f"\nEpoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Avg Val F1: {avg_val_f1:.4f}")

        if wandb.run:
            # Convert lists to numpy arrays for metric calculations
            np_val_olabels = np.array(all_val_olabels)
            np_val_oprobs = np.array(all_val_oprobs)
            np_val_opreds_binary = (np_val_oprobs > 0.5).astype(int)
            
            np_val_blabels = np.array(all_val_blabels)
            np_val_bprobs = np.array(all_val_bprobs)
            np_val_bpreds_binary = (np_val_bprobs > 0.5).astype(int)

            # Calculate all metrics for both teams
            val_prec_o = precision_score(np_val_olabels, np_val_opreds_binary, zero_division=0)
            val_recall_o = recall_score(np_val_olabels, np_val_opreds_binary, zero_division=0)
            
            val_prec_b = precision_score(np_val_blabels, np_val_bpreds_binary, zero_division=0)
            val_recall_b = recall_score(np_val_blabels, np_val_bpreds_binary, zero_division=0)

            # Prepare data for plots
            y_probas_orange_plots = np.stack([1 - np_val_oprobs, np_val_oprobs], axis=1)
            y_probas_blue_plots = np.stack([1 - np_val_bprobs, np_val_bprobs], axis=1)
            class_names = ["No Goal", "Goal"]

            # Log everything to W&B with standardized names
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": avg_train_loss,
                "val/loss": avg_val_loss,
                "val/f1_orange": val_f1_o,
                "val/f1_blue": val_f1_b,
                "val/avg_f1": avg_val_f1,
                "val/precision_orange": val_prec_o,
                "val/recall_orange": val_recall_o,
                "val/precision_blue": val_prec_b,
                "val/recall_blue": val_recall_b,
                "val/cm_orange": wandb.plot.confusion_matrix(y_true=np_val_olabels, preds=np_val_opreds_binary, class_names=class_names),
                "val/cm_blue": wandb.plot.confusion_matrix(y_true=np_val_blabels, preds=np_val_bpreds_binary, class_names=class_names),
                "val/pr_curve_orange": wandb.plot.pr_curve(y_true=np_val_olabels, y_probas=y_probas_orange_plots, labels=class_names),
                "val/pr_curve_blue": wandb.plot.pr_curve(y_true=np_val_blabels, y_probas=y_probas_blue_plots, labels=class_names),
                "val/roc_curve_orange": wandb.plot.roc_curve(y_true=np_val_olabels, y_probas=y_probas_orange_plots, labels=class_names),
                "val/roc_curve_blue": wandb.plot.roc_curve(y_true=np_val_blabels, y_probas=y_probas_blue_plots, labels=class_names)
            })

        current_wandb_id = wandb.run.id if wandb.run else None
        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1; best_model_path = os.path.join(os.path.dirname(args.checkpoint_path), 'best_gnn_model.pth')
            print(f"  *** New best model found (Avg F1: {best_val_f1:.4f}). Saving 'best' checkpoint. ***")
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'best_val_f1': best_val_f1, 'args': vars(args), 'wandb_run_id': current_wandb_id}, best_model_path)
        
        if (epoch + 1) % args.checkpoint_every == 0 or (epoch + 1) == args.epochs:
            print(f"--- Saving periodic checkpoint at epoch {epoch + 1}. ---")
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'best_val_f1': best_val_f1, 'args': vars(args), 'wandb_run_id': current_wandb_id}, args.checkpoint_path)

    print("\n--- Data Loading Summary ---")
    print(f"Total processed training samples: {len(train_dataset)}")
    print(f"Number of skipped/problematic training rows: {train_dataset.skipped_count}")
    
    print(f"Total processed validation samples: {len(val_dataset)}")
    print(f"Number of skipped/problematic validation rows: {val_dataset.skipped_count}")

    if wandb.run:
        wandb.finish()
    
    # ================= FINAL VALIDATION & TEST EVALUATION ============================
    print("\n--- Training Complete ---")
    best_model_path = os.path.join(os.path.dirname(args.checkpoint_path), 'best_gnn_model.pth') # Or a more specific name

    if not os.path.exists(best_model_path):
        print("--- No 'best' model checkpoint found. Skipping final test evaluation. ---")
    else:
        print(f"--- Loading best model from: {best_model_path} for final evaluation ---")
        checkpoint = torch.load(best_model_path, map_location=device)
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
        else:
            test_dataset = GraphLazyDataset(test_files)
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

            # Convert to numpy arrays
            y_true_o, y_prob_o = np.array(all_test_olabels), np.array(all_test_oprobs)
            y_true_b, y_prob_b = np.array(all_test_blabels), np.array(all_test_bprobs)

            # Apply optimal thresholds to get binary predictions
            preds_opt_o = (y_prob_o > optimal_threshold_orange).astype(int)
            preds_opt_b = (y_prob_b > optimal_threshold_blue).astype(int)

            # Calculate final metrics
            f1_opt_o = f1_score(y_true_o, preds_opt_o, zero_division=0)
            prec_opt_o = precision_score(y_true_o, preds_opt_o, zero_division=0)
            rec_opt_o = recall_score(y_true_o, preds_opt_o, zero_division=0)
            
            f1_opt_b = f1_score(y_true_b, preds_opt_b, zero_division=0)
            prec_opt_b = precision_score(y_true_b, preds_opt_b, zero_division=0)
            rec_opt_b = recall_score(y_true_b, preds_opt_b, zero_division=0)

            print("\n--- FINAL TEST RESULTS (with Optimal Thresholds) ---")
            print(f"  Orange Team -> F1: {f1_opt_o:.4f} | Precision: {prec_opt_o:.4f} | Recall: {rec_opt_o:.4f}")
            print(f"  Blue Team   -> F1: {f1_opt_b:.4f} | Precision: {prec_opt_b:.4f} | Recall: {rec_opt_b:.4f}")

            # --- Step 3: Log Final Metrics to W&B ---
            if wandb.run:
                print("\n--- Logging final test metrics to W&B ---")
                wandb.summary["best_epoch"] = checkpoint.get('epoch', 0) + 1
                wandb.summary["optimal_threshold_orange"] = optimal_threshold_orange
                wandb.summary["optimized_test_f1_orange"] = f1_opt_o
                wandb.summary["optimized_test_precision_orange"] = prec_opt_o
                wandb.summary["optimized_test_recall_orange"] = rec_opt_o
                wandb.summary["optimal_threshold_blue"] = optimal_threshold_blue
                wandb.summary["optimized_test_f1_blue"] = f1_opt_b
                wandb.summary["optimized_test_precision_blue"] = prec_opt_b
                wandb.summary["optimized_test_recall_blue"] = rec_opt_b

                class_names = ["No Goal", "Goal"]
                wandb.log({
                    "test/cm_orange_optimized": wandb.plot.confusion_matrix(y_true=y_true_o, preds=preds_opt_o, class_names=class_names),
                    "test/cm_blue_optimized": wandb.plot.confusion_matrix(y_true=y_true_b, preds=preds_opt_b, class_names=class_names),
                })

    print("\n--- Script Finished ---")

if __name__ == '__main__':
    main()