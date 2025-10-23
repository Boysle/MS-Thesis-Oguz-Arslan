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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (f1_score, precision_score, recall_score, 
                             confusion_matrix, precision_recall_curve)
import linecache

# ====================== CONFIGURATION & CONSTANTS ======================
NUM_PLAYERS = 6; PLAYER_FEATURES = 13; GLOBAL_FEATURES = 14
POS_MIN_X, POS_MAX_X = -4096, 4096; POS_MIN_Y, POS_MAX_Y = -6000, 6000; POS_MIN_Z, POS_MAX_Z = 0, 2044
VEL_MIN, VEL_MAX = -2300, 2300; BOOST_MIN, BOOST_MAX = 0, 100; BALL_VEL_MIN, BALL_VEL_MAX = -6000, 6000
BOOST_PAD_MIN, BOOST_PAD_MAX = 0, 10; DIST_MIN, DIST_MAX = 0, (8192**2 + 10240**2 + 2044**2)**0.5

def normalize(val, min_val, max_val): return (val - min_val) / (max_val - min_val + 1e-8)

# ====================== DATASET & MODEL CLASSES (DYNAMIC EDGE_DIM VERSION) ======================
class GraphLazyDataset(torch.utils.data.Dataset):
    def __init__(self, list_of_csv_paths, edge_dim):
        self.csv_paths = list_of_csv_paths
        self.edge_dim = edge_dim  # Store the required number of edge features
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
        dot = torch.sum(v1_norm * v2_norm, dim=-1); return (dot + 1) / 2

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
                feature_list = []
                inv_dist = 1.0 / (1.0 + (torch.linalg.norm(positions[i] - positions[j]) / 1500.0)**2)
                feature_list.append(inv_dist)
                if self.edge_dim >= 2:
                    feature_list.append(1.0 if teams[i] == teams[j] else 0.0)
                if self.edge_dim >= 3:
                    vec_i_to_j = positions[j] - positions[i]
                    vel_mag = torch.linalg.norm(velocities[i]); scaled_vel_mag = vel_mag / (2300.0 + 1e-8)
                    feature_list.append(self.normalized_dot_product(velocities[i], vec_i_to_j) * scaled_vel_mag)
                if self.edge_dim == 4:
                    if self.edge_dim < 3: vec_i_to_j = positions[j] - positions[i]
                    feature_list.append(self.normalized_dot_product(forwards[i], vec_i_to_j))
                edge_attrs.append(feature_list)
            edge_attr_tensor = torch.tensor(edge_attrs, dtype=torch.float32)

            return Data(x=x_tensor, edge_index=self.edge_index, edge_attr=edge_attr_tensor, global_features=global_tensor, y_orange=orange_y, y_blue=blue_y)
        except (ValueError, KeyError, IndexError): return self.empty_data()

    def empty_data(self):
        x_tensor = torch.zeros((NUM_PLAYERS, PLAYER_FEATURES), dtype=torch.float32)
        global_tensor = torch.zeros((1, GLOBAL_FEATURES), dtype=torch.float32)
        edge_attr_tensor = torch.zeros((self.edge_index.size(1), self.edge_dim), dtype=torch.float32)
        return Data(x=x_tensor, edge_index=self.edge_index, edge_attr=edge_attr_tensor, global_features=global_tensor, y_orange=torch.tensor([0.0]), y_blue=torch.tensor([0.0]))

def collate_fn_master(batch):
    batch = [item for item in batch if item is not None];
    if not batch: return None
    return Batch.from_data_list(batch)

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

# ====================== HELPER & PLOTTING FUNCTIONS ======================
def find_optimal_threshold(y_true, y_pred_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_f1_idx = np.argmax(f1_scores[:-1]); return thresholds[best_f1_idx], f1_scores[best_f1_idx]

def get_all_predictions(model, loader, device, desc=""):
    model.eval(); all_oprobs, all_olabels, all_bprobs, all_blabels = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            if batch is None: continue
            batch = batch.to(device); o_logits, b_logits = model(batch)
            all_oprobs.extend(torch.sigmoid(o_logits).cpu().numpy().flatten())
            all_olabels.extend(batch.y_orange.cpu().numpy().flatten())
            all_bprobs.extend(torch.sigmoid(b_logits).cpu().numpy().flatten())
            all_blabels.extend(batch.y_blue.cpu().numpy().flatten())
    return np.array(all_olabels), np.array(all_oprobs), np.array(all_blabels), np.array(all_bprobs)

def plot_and_save_distribution(y_true, y_prob, threshold, model_name, team_name, output_dir):
    plt.figure(figsize=(12, 8))
    sns.histplot(y_prob[y_true == 0], bins=50, color='skyblue', label='Actual: No Goal', kde=True, log_scale=(False, True))
    sns.histplot(y_prob[y_true == 1], bins=50, color='salmon', label='Actual: Goal', kde=True, log_scale=(False, True))
    plt.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Default Threshold (0.5)')
    plt.axvline(threshold, color='red', linestyle='-', linewidth=2, label=f'Optimal Threshold ({threshold:.4f})')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title(f'Log-Scale Prob. Distribution ({model_name} - {team_name.upper()})', fontsize=16)
    plt.xlabel('Predicted Probability of Goal'); plt.ylabel('Frequency (Log Scale)'); plt.legend()
    filepath = os.path.join(output_dir, f'{model_name}_{team_name}_distribution.png')
    plt.savefig(filepath, bbox_inches='tight'); plt.close()
    print(f"  > Distribution plot saved to: {filepath}")

def plot_and_save_confusion_matrix(y_true, y_pred, threshold, model_name, team_name, output_dir):
    cm = confusion_matrix(y_true, y_pred); plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Goal", "Goal"], yticklabels=["No Goal", "Goal"])
    plt.title(f'Confusion Matrix ({model_name} - {team_name.upper()}) at Thresh={threshold:.2f}')
    plt.ylabel('Actual Label'); plt.xlabel('Predicted Label'); thresh_str = str(f"{threshold:.4f}").replace('.', 'p')
    filepath = os.path.join(output_dir, f'{model_name}_{team_name}_cm_thresh_{thresh_str}.png')
    plt.savefig(filepath, bbox_inches='tight'); plt.close()
    print(f"  > Confusion matrix saved to: {filepath}")

def plot_and_save_metrics_table(metrics_default, metrics_optimized, threshold, model_name, team_name, output_dir):
    metrics_data = [[f"{metrics_default[0]:.4f}", f"{metrics_optimized[0]:.4f}"], [f"{metrics_default[1]:.4f}", f"{metrics_optimized[1]:.4f}"], [f"{metrics_default[2]:.4f}", f"{metrics_optimized[2]:.4f}"]]
    fig, ax = plt.subplots(figsize=(10, 2)); ax.axis('tight'); ax.axis('off')
    table = ax.table(cellText=metrics_data, rowLabels=['Precision', 'Recall', 'F1-Score'], colLabels=[f'Default (0.5)', f'Optimal ({threshold:.4f})'], cellLoc='center', loc='center', colWidths=[0.3, 0.3])
    table.auto_set_font_size(False); table.set_fontsize(12); table.scale(1.2, 1.2)
    plt.title(f'Performance Metrics ({model_name} - {team_name.upper()})', fontsize=14, pad=20)
    filepath = os.path.join(output_dir, f'{model_name}_{team_name}_metrics_table.png')
    plt.savefig(filepath, bbox_inches='tight'); plt.close()
    print(f"  > Metrics table saved to: {filepath}")

# ====================== ARGUMENT PARSER ======================
def parse_args():
    parser = argparse.ArgumentParser(description="Local Analysis Script for GAT Ablation Models")
    parser.add_argument('--model-paths', type=str, nargs='+', required=True, help='List of paths to the trained model checkpoints (.pth).')
    parser.add_argument('--model-names', type=str, nargs='+', required=True, help='A parallel list of short names for analysis (e.g., DistOnly, DistTeam).')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to the root of the split dataset.')
    parser.add_argument('--output-dir', type=str, default='./analysis_results_ablation', help='Directory to save all output images.')
    parser.add_argument('--batch-size', type=int, default=2048, help='Batch size for evaluation.')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of workers for DataLoader. Set to 0 for Windows.')
    return parser.parse_args()

# ====================== MAIN ANALYSIS SCRIPT ======================
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")
    if len(args.model_paths) != len(args.model_names): raise ValueError("Mismatch between --model-paths and --model-names.")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"--- All plots will be saved in: {os.path.abspath(args.output_dir)} ---")
    
    val_files = [os.path.join(args.data_dir, 'val', f) for f in os.listdir(os.path.join(args.data_dir, 'val')) if f.endswith('.csv')]
    test_files = [os.path.join(args.data_dir, 'test', f) for f in os.listdir(os.path.join(args.data_dir, 'test')) if f.endswith('.csv')]

    for model_path, model_name in zip(args.model_paths, args.model_names):
        print(f"\n\n{'='*30}\n--- Analyzing Model: {model_name} ---\n{'='*30}")
        
        # ========================= CORRECTED LOGIC HERE =========================
        # Clean the model name to be robust to hyphens or casing
        model_name_clean = model_name.lower().replace('-', '')

        # Check for the most specific names first
        if "allfeatures" in model_name_clean:
            edge_dim = 4
        elif "distteamvelo" in model_name_clean:
            edge_dim = 3
        elif "distteam" in model_name_clean:
            edge_dim = 2
        elif "dist" in model_name_clean:
            edge_dim = 1
        else:
            raise ValueError(f"Could not determine edge_dim from model name: {model_name}. Use names like 'DistOnly', 'DistTeam', 'AllFeatures'.")
        # ========================================================================
        
        print(f"  > Inferred edge_dim: {edge_dim}")
        
        print("  > Initializing DataLoaders with the correct edge dimension...")
        val_dataset = GraphLazyDataset(val_files, edge_dim=edge_dim)
        test_dataset = GraphLazyDataset(test_files, edge_dim=edge_dim)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn_master)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn_master)

        model = RocketLeagueGAT(PLAYER_FEATURES, GLOBAL_FEATURES, hidden_dim=64, edge_dim=edge_dim).to(device)
        checkpoint = torch.load(model_path, map_location=device); model.load_state_dict(checkpoint['model_state'])
        
        val_olabels, val_oprobs, val_blabels, val_bprobs = get_all_predictions(model, val_loader, device, desc=f"[{model_name} VAL]")
        test_olabels, test_oprobs, test_blabels, test_bprobs = get_all_predictions(model, test_loader, device, desc=f"[{model_name} TEST]")
        
        opt_thresh_o, _ = find_optimal_threshold(val_olabels, val_oprobs)
        opt_thresh_b, _ = find_optimal_threshold(val_blabels, val_bprobs)
        print(f"\n  > Optimal Thresholds -> Orange: {opt_thresh_o:.4f}, Blue: {opt_thresh_b:.4f}")

        team_map = {'orange': [test_olabels, test_oprobs, opt_thresh_o], 'blue': [test_blabels, test_bprobs, opt_thresh_b]}
        for team, (labels, probs, thresh) in team_map.items():
            print(f"\n--- Analysis for {team.upper()} Team ---")
            team_output_dir = os.path.join(args.output_dir, model_name); os.makedirs(team_output_dir, exist_ok=True)
            preds_def = (probs > 0.5).astype(int); preds_opt = (probs > thresh).astype(int)
            metrics_def = [precision_score(labels, preds_def, zero_division=0), recall_score(labels, preds_def, zero_division=0), f1_score(labels, preds_def, zero_division=0)]
            metrics_opt = [precision_score(labels, preds_opt, zero_division=0), recall_score(labels, preds_opt, zero_division=0), f1_score(labels, preds_opt, zero_division=0)]
            
            plot_and_save_distribution(labels, probs, thresh, model_name, team, team_output_dir)
            plot_and_save_confusion_matrix(labels, preds_def, 0.5, model_name, team, team_output_dir)
            plot_and_save_confusion_matrix(labels, preds_opt, thresh, model_name, team, team_output_dir)
            plot_and_save_metrics_table(metrics_def, metrics_opt, thresh, model_name, team, team_output_dir)
            
            print("\n--- TEST SET METRICS ---")
            print(f"  Results for {model_name.upper()} - {team.upper()} Team:")
            print(f"    Default (0.5) Threshold -> Precision: {metrics_def[0]:.4f}, Recall: {metrics_def[1]:.4f}, F1-Score: {metrics_def[2]:.4f}")
            print(f"    Optimal ({thresh:.4f}) Threshold -> Precision: {metrics_opt[0]:.4f}, Recall: {metrics_opt[1]:.4f}, F1-Score: {metrics_opt[2]:.4f}")

    print("\n--- Analysis Complete ---")

if __name__ == '__main__':
    main()