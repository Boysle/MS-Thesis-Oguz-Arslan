import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (f1_score, precision_score, recall_score, 
                             confusion_matrix, precision_recall_curve, roc_curve, auc)
import linecache

# ====================== CONFIGURATION & CONSTANTS ======================
NUM_PLAYERS = 6
PLAYER_FEATURES = 13
GLOBAL_FEATURES = 14

# Normalization bounds
POS_MIN_X, POS_MAX_X = -4096, 4096; POS_MIN_Y, POS_MAX_Y = -6000, 6000; POS_MIN_Z, POS_MAX_Z = 0, 2044
VEL_MIN, VEL_MAX = -2300, 2300; BOOST_MIN, BOOST_MAX = 0, 100; BALL_VEL_MIN, BALL_VEL_MAX = -6000, 6000
BOOST_PAD_MIN, BOOST_PAD_MAX = 0, 10; DIST_MIN, DIST_MAX = 0, (8192**2 + 10240**2 + 2044**2)**0.5

def normalize(val, min_val, max_val):
    return (val - min_val) / (max_val - min_val + 1e-8)

# ====================== DATASET CLASS (LAZY LOADING FOR GRAPHS) ======================
class GraphLazyDataset(torch.utils.data.Dataset):
    def __init__(self, list_of_csv_paths):
        self.csv_paths = list_of_csv_paths
        self.file_info, self.cumulative_rows, self.header, total_rows = [], [0], None, 0
        desc = f"Indexing {os.path.basename(os.path.dirname(list_of_csv_paths[0]))} files"
        for path in tqdm(self.csv_paths, desc=desc):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    if self.header is None: self.header = f.readline().strip().split(',')
                    num_lines = sum(1 for _ in f)
                if num_lines > 0:
                    self.file_info.append({'path': path, 'rows': num_lines})
                    total_rows += num_lines; self.cumulative_rows.append(total_rows)
            except Exception as e: print(f"\nWarning: Could not process file {path}. Error: {e}")
        self.length = total_rows
        self.edge_index = torch.tensor([(i, j) for i in range(NUM_PLAYERS) for j in range(NUM_PLAYERS) if i != j], dtype=torch.long).t().contiguous()

    def normalized_dot_product(self, v1, v2):
        v1_norm = v1 / (torch.linalg.norm(v1, dim=-1, keepdim=True) + 1e-8)
        v2_norm = v2 / (torch.linalg.norm(v2, dim=-1, keepdim=True) + 1e-8)
        dot = torch.sum(v1_norm * v2_norm, dim=-1)
        return (dot + 1) / 2

    def __len__(self): return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length: raise IndexError("Index out of range")
        file_index = np.searchsorted(self.cumulative_rows, idx, side='right') - 1
        file_path = self.file_info[file_index]['path']; local_idx = idx - self.cumulative_rows[file_index]
        line = linecache.getline(file_path, local_idx + 2)
        try:
            if not line.strip(): return self.empty_data()
            row = dict(zip(self.header, line.strip().split(',')))
            x_features = [[normalize(float(row[f'p{i}_pos_x']), POS_MIN_X, POS_MAX_X), normalize(float(row[f'p{i}_pos_y']), POS_MIN_Y, POS_MAX_Y), normalize(float(row[f'p{i}_pos_z']), POS_MIN_Z, POS_MAX_Z), normalize(float(row[f'p{i}_vel_x']), VEL_MIN, VEL_MAX), normalize(float(row[f'p{i}_vel_y']), VEL_MIN, VEL_MAX), normalize(float(row[f'p{i}_vel_z']), VEL_MIN, VEL_MAX), float(row[f'p{i}_forward_x']), float(row[f'p{i}_forward_y']), float(row[f'p{i}_forward_z']), normalize(float(row[f'p{i}_boost_amount']), BOOST_MIN, BOOST_MAX), float(row[f'p{i}_team']), float(row[f'p{i}_alive']), normalize(float(row[f'p{i}_dist_to_ball']), DIST_MIN, DIST_MAX)] for i in range(NUM_PLAYERS)]
            x_tensor = torch.tensor(x_features, dtype=torch.float32)
            global_features = [normalize(float(row['ball_pos_x']), POS_MIN_X, POS_MAX_X), normalize(float(row['ball_pos_y']), POS_MIN_Y, POS_MAX_Y), normalize(float(row['ball_pos_z']), POS_MIN_Z, POS_MAX_Z), normalize(float(row['ball_vel_x']), BALL_VEL_MIN, BALL_VEL_MAX), normalize(float(row['ball_vel_y']), BALL_VEL_MIN, BALL_VEL_MAX), normalize(float(row['ball_vel_z']), BALL_VEL_MIN, BALL_VEL_MAX)] + [normalize(float(row[f'boost_pad_{i}_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX) for i in range(6)] + [float(row['ball_hit_team_num']), normalize(min(float(row['seconds_remaining']), 300.0), 0, 300)]
            global_tensor = torch.tensor(global_features, dtype=torch.float32).unsqueeze(0)
            orange_y = torch.tensor([float(row['team_1_goal_in_event_window'])], dtype=torch.float32)
            blue_y = torch.tensor([float(row['team_0_goal_in_event_window'])], dtype=torch.float32)
            positions = x_tensor[:, 0:3]; velocities = x_tensor[:, 3:6]; forwards = x_tensor[:, 6:9]; teams = x_tensor[:, 10]
            edge_attrs = []
            for i, j in self.edge_index.t():
                dist = torch.linalg.norm(positions[i] - positions[j]); inv_dist = 1.0 / (1.0 + (dist / 1500.0)**2)
                same_team = 1.0 if teams[i] == teams[j] else 0.0
                vec_i_to_j = positions[j] - positions[i]
                vel_mag = torch.linalg.norm(velocities[i]); scaled_vel_mag = vel_mag / (2300.0 + 1e-8)
                vel_align = self.normalized_dot_product(velocities[i], vec_i_to_j); velocity_feature = vel_align * scaled_vel_mag
                forward_align = self.normalized_dot_product(forwards[i], vec_i_to_j)
                edge_attrs.append([inv_dist, same_team, velocity_feature, forward_align])
            edge_attr_tensor = torch.tensor(edge_attrs, dtype=torch.float32)
            return Data(x=x_tensor, edge_index=self.edge_index, edge_attr=edge_attr_tensor, global_features=global_tensor, y_orange=orange_y, y_blue=blue_y)
        except (ValueError, KeyError, IndexError): return self.empty_data()

    def empty_data(self):
        x_tensor = torch.zeros((NUM_PLAYERS, PLAYER_FEATURES), dtype=torch.float32)
        global_tensor = torch.zeros((1, GLOBAL_FEATURES), dtype=torch.float32)
        edge_attr_tensor = torch.zeros((self.edge_index.size(1), 4), dtype=torch.float32)
        return Data(x=x_tensor, edge_index=self.edge_index, edge_attr=edge_attr_tensor, global_features=global_tensor, y_orange=torch.tensor([0.0]), y_blue=torch.tensor([0.0]))

def collate_fn_master(batch):
    batch = [item for item in batch if item is not None];
    if not batch: return None
    return Batch.from_data_list(batch)

# ====================== MODEL ARCHITECTURE ======================
class RocketLeagueGAT(nn.Module):
    def __init__(self, player_features, global_features, hidden_dim, edge_dim):
        super().__init__()
        self.conv1 = GATConv(player_features, hidden_dim, heads=4, edge_dim=edge_dim, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 4)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, edge_dim=edge_dim, concat=False)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.orange_head = nn.Sequential(nn.Linear(hidden_dim + global_features, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))
        self.blue_head = nn.Sequential(nn.Linear(hidden_dim + global_features, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_attr)))
        graph_embed = global_mean_pool(x, data.batch)
        combined = torch.cat([graph_embed, data.global_features], dim=1)
        return self.orange_head(combined), self.blue_head(combined)

# ====================== HELPER & PLOTTING FUNCTIONS (CONSISTENT VERSION) ======================
def find_optimal_threshold(y_true, y_pred_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_f1_idx = np.argmax(f1_scores[:-1])
    return thresholds[best_f1_idx], f1_scores[best_f1_idx]

def plot_distribution(y_true, y_prob, threshold, model_name, team_name):
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.histplot(y_prob[y_true == 0], bins=50, color='skyblue', label='Actual: No Goal', kde=True, log_scale=(False, True), ax=ax)
    sns.histplot(y_prob[y_true == 1], bins=50, color='salmon', label='Actual: Goal', kde=True, log_scale=(False, True), ax=ax)
    ax.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Default Threshold (0.5)')
    ax.axvline(threshold, color='red', linestyle='-', linewidth=2, label=f'Optimal Threshold ({threshold:.4f})')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_title(f'Log-Scale Prob. Distribution ({model_name} - {team_name.upper()})', fontsize=16)
    ax.set_xlabel('Predicted Probability of Goal'); ax.set_ylabel('Frequency (Log Scale)'); ax.legend()
    plt.tight_layout(); return fig

def plot_confusion_matrix(y_true, y_pred, threshold, model_name, team_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=["No Goal", "Goal"], yticklabels=["No Goal", "Goal"])
    ax.set_xlabel('Predicted Label'); ax.set_ylabel('True Label')
    ax.set_title(f'Confusion Matrix ({model_name} - {team_name.upper()}) at Thresh={threshold:.2f}')
    plt.tight_layout(); return fig

def plot_metrics_table(metrics_default, metrics_optimized, threshold, model_name, team_name):
    metrics_data = [[f"{metrics_default[0]:.4f}", f"{metrics_optimized[0]:.4f}"], [f"{metrics_default[1]:.4f}", f"{metrics_optimized[1]:.4f}"], [f"{metrics_default[2]:.4f}", f"{metrics_optimized[2]:.4f}"]]
    fig, ax = plt.subplots(figsize=(10, 2)); ax.axis('tight'); ax.axis('off')
    table = ax.table(cellText=metrics_data, rowLabels=['Precision', 'Recall', 'F1-Score'], colLabels=[f'Default (0.5)', f'Optimal ({threshold:.4f})'], cellLoc='center', loc='center', colWidths=[0.3, 0.3])
    table.auto_set_font_size(False); table.set_fontsize(12); table.scale(1.2, 1.2)
    plt.title(f'Performance Metrics ({model_name} - {team_name.upper()})', fontsize=14, pad=20)
    plt.tight_layout(); return fig

def plot_pr_curve(y_true, y_prob, model_name, team_name):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, marker='.'); ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title(f"PR Curve ({model_name} - {team_name.upper()})"); ax.grid(True); plt.tight_layout(); return fig

def get_all_predictions(model, loader, device, desc=""):
    model.eval()
    all_oprobs, all_olabels, all_bprobs, all_blabels = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            if batch is None: continue
            batch = batch.to(device)
            o_logits, b_logits = model(batch)
            all_oprobs.extend(torch.sigmoid(o_logits).cpu().numpy().flatten())
            all_olabels.extend(batch.y_orange.cpu().numpy().flatten())
            all_bprobs.extend(torch.sigmoid(b_logits).cpu().numpy().flatten())
            all_blabels.extend(batch.y_blue.cpu().numpy().flatten())
    return np.array(all_olabels), np.array(all_oprobs), np.array(all_blabels), np.array(all_bprobs)

# ====================== ARGUMENT PARSER ======================
def parse_args():
    parser = argparse.ArgumentParser(description="Consistent Analysis Script for GAT Ablation Models")
    parser.add_argument('--model-paths', type=str, nargs='+', required=True, help='List of paths to the trained model checkpoints (.pth).')
    parser.add_argument('--model-names', type=str, nargs='+', required=True, help='A parallel list of short names for logging (e.g., DistOnly, DistTeam).')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to the root of the split dataset.')
    parser.add_argument('--batch-size', type=int, default=2048, help='Batch size for evaluation.')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of workers for DataLoader.')
    parser.add_argument('--wandb-project', type=str, default="rl-goal-prediction-gnn-analysis", help="W&B project name.")
    parser.add_argument('--run-name', type=str, default="Ablation-Analysis", help="Custom name for the W&B run.")
    return parser.parse_args()

# ====================== MAIN ANALYSIS SCRIPT ======================
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")
    if len(args.model_paths) != len(args.model_names): raise ValueError("Mismatch between number of --model-paths and --model-names.")
    try:
        wandb.init(project=args.wandb_project, name=args.run_name, config=args)
    except Exception as e:
        print(f"--- Could not initialize W&B: {e}. Running without logging. ---"); wandb.run = None
    
    print("\n--- Initializing Data Loaders (This may take a moment) ---")
    val_files = [os.path.join(args.data_dir, 'val', f) for f in os.listdir(os.path.join(args.data_dir, 'val')) if f.endswith('.csv')]
    test_files = [os.path.join(args.data_dir, 'test', f) for f in os.listdir(os.path.join(args.data_dir, 'test')) if f.endswith('.csv')]
    val_dataset = GraphLazyDataset(val_files); test_dataset = GraphLazyDataset(test_files)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn_master)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn_master)

    for model_path, model_name in zip(args.model_paths, args.model_names):
        print(f"\n\n{'='*30}\n--- Analyzing Model: {model_name} ---\n{'='*30}")
        if "dist-team-velo" in model_path: edge_dim = 3
        elif "dist-team" in model_path: edge_dim = 2
        elif "dist" in model_path: edge_dim = 1
        elif "all-features" in model_path: edge_dim = 4
        else: raise ValueError(f"Could not determine edge_dim from path: {model_path}")
        print(f"  > Inferred edge_dim: {edge_dim}")
        
        model = RocketLeagueGAT(PLAYER_FEATURES, GLOBAL_FEATURES, hidden_dim=64, edge_dim=edge_dim).to(device)
        checkpoint = torch.load(model_path, map_location=device); model.load_state_dict(checkpoint['model_state'])
        
        val_olabels, val_oprobs, val_blabels, val_bprobs = get_all_predictions(model, val_loader, device, desc=f"[{model_name} VAL]")
        test_olabels, test_oprobs, test_blabels, test_bprobs = get_all_predictions(model, test_loader, device, desc=f"[{model_name} TEST]")
        
        opt_thresh_o, _ = find_optimal_threshold(val_olabels, val_oprobs)
        opt_thresh_b, _ = find_optimal_threshold(val_blabels, val_bprobs)
        print(f"  > Optimal Thresholds -> Orange: {opt_thresh_o:.4f}, Blue: {opt_thresh_b:.4f}")

        if wandb.run:
            log_dict = {}; team_map = {'orange': [test_olabels, test_oprobs, opt_thresh_o], 'blue': [test_blabels, test_bprobs, opt_thresh_b]}
            for team, (labels, probs, thresh) in team_map.items():
                print(f"  > Analyzing and plotting for {team.upper()} team...")
                preds_def = (probs > 0.5).astype(int); preds_opt = (probs > thresh).astype(int)
                metrics_def = [precision_score(labels, preds_def, zero_division=0), recall_score(labels, preds_def, zero_division=0), f1_score(labels, preds_def, zero_division=0)]
                metrics_opt = [precision_score(labels, preds_opt, zero_division=0), recall_score(labels, preds_opt, zero_division=0), f1_score(labels, preds_opt, zero_division=0)]
                
                dist_fig = plot_distribution(labels, probs, thresh, model_name, team.capitalize()); log_dict[f"{model_name}/plots/distribution_{team}"] = wandb.Image(dist_fig); plt.close(dist_fig)
                cm_def_fig = plot_confusion_matrix(labels, preds_def, 0.5, model_name, team.capitalize()); log_dict[f"{model_name}/plots/cm_default_{team}"] = wandb.Image(cm_def_fig); plt.close(cm_def_fig)
                cm_opt_fig = plot_confusion_matrix(labels, preds_opt, thresh, model_name, team.capitalize()); log_dict[f"{model_name}/plots/cm_optimized_{team}"] = wandb.Image(cm_opt_fig); plt.close(cm_opt_fig)
                table_fig = plot_metrics_table(metrics_def, metrics_opt, thresh, model_name, team.capitalize()); log_dict[f"{model_name}/plots/metrics_table_{team}"] = wandb.Image(table_fig); plt.close(table_fig)
                pr_fig = plot_pr_curve(labels, probs, model_name, team.capitalize()); log_dict[f"{model_name}/plots/pr_curve_{team}"] = wandb.Image(pr_fig); plt.close(pr_fig)
                
                log_dict[f"{model_name}/z_metrics/f1_optimized_{team}"] = metrics_opt[2]
                log_dict[f"{model_name}/z_metrics/precision_optimized_{team}"] = metrics_opt[0]
                log_dict[f"{model_name}/z_metrics/recall_optimized_{team}"] = metrics_opt[1]
                log_dict[f"{model_name}/z_metrics/f1_default_{team}"] = metrics_def[2]
                log_dict[f"{model_name}/z_metrics/optimal_threshold_{team}"] = thresh
                wandb.summary[f"{model_name}_f1_{team}"] = metrics_opt[2]
            
            wandb.log(log_dict)

    if wandb.run: wandb.finish()
    print("\n--- Analysis Complete ---")

if __name__ == '__main__':
    main()