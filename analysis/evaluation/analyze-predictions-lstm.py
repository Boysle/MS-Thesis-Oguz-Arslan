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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, precision_score, recall_score, precision_recall_curve, 
    confusion_matrix, accuracy_score, average_precision_score, log_loss
)

# ====================== CONFIGURATION & CONSTANTS ======================
NUM_PLAYERS = 6
PLAYER_FEATURES = 13
GLOBAL_FEATURES = 14 
TOTAL_FLAT_FEATURES = (NUM_PLAYERS * PLAYER_FEATURES) + GLOBAL_FEATURES
SEQUENCE_LENGTH = 6 # 5 previous + 1 current

# --- Normalization (Universal) ---
POS_MIN_X, POS_MAX_X = -4096, 4096; POS_MIN_Y, POS_MAX_Y = -6000, 6000; POS_MIN_Z, POS_MAX_Z = 0, 2044
VEL_MIN, VEL_MAX = -2300, 2300; BOOST_MIN, BOOST_MAX = 0, 100; BALL_VEL_MIN, BALL_VEL_MAX = -6000, 6000
BOOST_PAD_MIN, BOOST_PAD_MAX = 0, 10; DIST_MIN, DIST_MAX = 0, (8192**2 + 10240**2 + 2044**2)**0.5

def normalize(val, min_val, max_val):
    return (val - min_val) / (max_val - min_val + 1e-8)

# ====================== DATASET CLASS (Copied from lstm_train_v5.py) ======================
class SequentialLazyDataset(Dataset):
    def __init__(self, list_of_csv_paths, sequence_length=6):
        self.csv_paths = list_of_csv_paths
        self.sequence_length = sequence_length
        self.file_info, self.cumulative_rows, self.header, total_rows = [], [0], None, 0
        
        desc = f"Indexing {os.path.basename(os.path.dirname(list_of_csv_paths[0]))} files"
        for path in tqdm(self.csv_paths, desc=desc):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    if self.header is None: 
                        self.header = f.readline().strip().split(',')
                        # Build the exact 92-feature order once
                        self.player_cols = [f'p{i}_{feat}' for i in range(NUM_PLAYERS) for feat in ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'forward_x', 'forward_y', 'forward_z', 'boost_amount', 'team', 'alive', 'dist_to_ball']]
                        self.global_cols = ['ball_pos_x', 'ball_pos_y', 'ball_pos_z', 'ball_vel_x', 'ball_vel_y', 'ball_vel_z', 'boost_pad_0_respawn', 'boost_pad_1_respawn', 'boost_pad_2_respawn', 'boost_pad_3_respawn', 'boost_pad_4_respawn', 'boost_pad_5_respawn', 'ball_hit_team_num', 'seconds_remaining']
                        self.feature_cols_ordered = self.player_cols + self.global_cols
                        
                    num_lines = sum(1 for _ in f)
                if num_lines > 0: 
                    self.file_info.append({'path': path, 'rows': num_lines})
                    total_rows += num_lines
                    self.cumulative_rows.append(total_rows)
            except Exception as e: 
                print(f"\nWarning: Could not process file {path}. Skipping. Error: {e}")
        
        self.length = total_rows
        self.start_index = self.sequence_length - 1 
        self.effective_length = self.length - self.start_index
        
        print(f"\n--- Indexing complete. Total rows: {self.length}. Effective samples: {self.effective_length} ---")

    def __len__(self):
        return self.effective_length

    def _get_row(self, idx):
        if idx < 0 or idx >= self.length:
            return None, -1 
        file_index = np.searchsorted(self.cumulative_rows, idx, side='right') - 1
        file_path = self.file_info[file_index]['path']
        local_idx = idx - self.cumulative_rows[file_index]
        line = linecache.getline(file_path, local_idx + 2)
        if not line.strip():
            return None, file_index
        try:
            row = dict(zip(self.header, line.strip().split(',')))
            return row, file_index
        except (ValueError, KeyError, IndexError):
            return None, file_index

    def _normalize_and_flatten(self, row):
        features = []
        try:
            for col in self.feature_cols_ordered:
                val = float(row[col])
                if 'pos_x' in col or 'ball_pos_x' in col: features.append(normalize(val, POS_MIN_X, POS_MAX_X))
                elif 'pos_y' in col or 'ball_pos_y' in col: features.append(normalize(val, POS_MIN_Y, POS_MAX_Y))
                elif 'pos_z' in col or 'ball_pos_z' in col: features.append(normalize(val, POS_MIN_Z, POS_MAX_Z))
                elif 'vel' in col and 'ball' not in col: features.append(normalize(val, VEL_MIN, VEL_MAX))
                elif 'ball_vel' in col: features.append(normalize(val, BALL_VEL_MIN, BALL_VEL_MAX))
                elif 'boost_amount' in col: features.append(normalize(val, BOOST_MIN, BOOST_MAX))
                elif 'dist_to_ball' in col: features.append(normalize(val, DIST_MIN, DIST_MAX))
                elif 'boost_pad' in col: features.append(normalize(val, BOOST_PAD_MIN, BOOST_PAD_MAX))
                elif 'seconds_remaining' in col: features.append(normalize(min(val, 300.0), 0, 300))
                else: features.append(val)
            return torch.tensor(features, dtype=torch.float32)
        except Exception as e:
            return None

    def __getitem__(self, idx):
        current_idx = idx + self.start_index
        sequence_rows = []
        outlier_count_replay, outlier_count_score, outlier_count_overtime = 0, 0, 0
        
        anchor_row, anchor_file_idx = self._get_row(current_idx)
        if anchor_row is None:
            return None
        
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
                sequence_rows.insert(0, prev_row)
                last_valid_row = prev_row
            else:
                sequence_rows.insert(0, last_valid_row)

        x_seq_tensors = []
        for row in sequence_rows:
            features = self._normalize_and_flatten(row)
            if features is None: return None
            x_seq_tensors.append(features)
        
        x_tensor = torch.stack(x_seq_tensors)
        y_orange = torch.tensor([float(anchor_row['team_1_goal_in_event_window'])], dtype=torch.float32)
        y_blue = torch.tensor([float(anchor_row['team_0_goal_in_event_window'])], dtype=torch.float32)

        return x_tensor, y_orange, y_blue, outlier_count_replay, outlier_count_score, outlier_count_overtime

# ====================== LSTM MODEL (Copied from lstm_train_v5.py) ======================
class BaselineLSTM(nn.Module):
    def __init__(self, input_dim=TOTAL_FLAT_FEATURES, hidden_dim=128, num_layers=1, dropout=0.3):
        super(BaselineLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0  
        )
        self.dropout = nn.Dropout(p=dropout)
        self.orange_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.blue_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x_seq):
        lstm_out, (hn, cn) = self.lstm(x_seq)
        last_out = lstm_out[:, -1, :]
        last_out = self.dropout(last_out)
        orange_logits = self.orange_head(last_out)
        blue_logits = self.blue_head(last_out)
        return orange_logits, blue_logits

# ====================== HELPER FUNCTIONS ======================

def collate_fn_master(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None
    x_tensors, y_oranges, y_blues, replay_outliers, score_outliers, overtime_outliers = zip(*batch)
    x_batch = torch.stack(x_tensors)
    y_o_batch = torch.stack(y_oranges)
    y_b_batch = torch.stack(y_blues)
    total_replay_outliers = sum(replay_outliers)
    total_score_outliers = sum(score_outliers)
    total_overtime_outliers = sum(overtime_outliers)
    return x_batch, y_o_batch, y_b_batch, total_replay_outliers, total_score_outliers, total_overtime_outliers

def calculate_class_weights(train_files):
    print("\n--- Calculating class weights for loss function ---")
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
    return pos_weight_orange, pos_weight_blue

def get_predictions_and_loss_sequential(model, loader, device, criterion_o, criterion_b):
    model.eval()
    all_orange_labels, all_blue_labels = [], []
    all_orange_probs, all_blue_probs = [], []
    total_loss_o, total_loss_b = 0.0, 0.0
    total_replay_outliers, total_score_outliers, total_overtime_outliers = 0, 0, 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Getting predictions and loss"):
            if batch is None: continue
            x_batch, y_o_batch, y_b_batch, replay_outliers, score_outliers, overtime_outliers = batch
            x_batch, y_o_batch, y_b_batch = x_batch.to(device), y_o_batch.to(device), y_b_batch.to(device)
            
            orange_logits, blue_logits = model(x_batch)
            total_loss_o += criterion_o(orange_logits, y_o_batch).item() * x_batch.size(0)
            total_loss_b += criterion_b(blue_logits, y_b_batch).item() * x_batch.size(0)
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

# --- Plotting Functions ---
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

# ====================== MASTER ANALYSIS FUNCTION ======================
def run_full_analysis(model, loader, device, pos_weight_orange, pos_weight_blue, team, threshold, set_name, output_dir):
    """
    Runs the complete analysis suite for a given dataset (val or test).
    """
    print(f"\n{'='*25} Starting Analysis for {set_name.upper()} Set {'='*25}")
    
    # --- 1. Get Predictions & Loss ---
    (labels_o, labels_b, probs_o, probs_b, 
     loss_o, loss_b, replay_outliers, 
     score_outliers, overtime_outliers) = get_predictions_and_loss_sequential(
         model, loader, device, 
         nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_orange]).to(device)),
         nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_blue]).to(device))
     )
    
    print(f"  Outliers Found in {set_name} set: Replay={replay_outliers}, Score={score_outliers}, Overtime={overtime_outliers}")
    
    # Select the correct team's data
    y_true = labels_o if team == 'orange' else labels_b
    y_prob = probs_o if team == 'orange' else probs_b
    weighted_loss = loss_o if team == 'orange' else loss_b

    # --- 2. Calculate All Metrics ---
    # Default @ 0.5
    preds_def = (y_prob > 0.5).astype(int)
    tn_def, fp_def, fn_def, tp_def = confusion_matrix(y_true, preds_def, labels=[0,1]).ravel()
    f1_def = f1_score(y_true, preds_def, zero_division=0)
    prec_def = precision_score(y_true, preds_def, zero_division=0)
    rec_def = recall_score(y_true, preds_def, zero_division=0)
    acc_def = accuracy_score(y_true, preds_def)

    # Custom Threshold
    preds_opt = (y_prob > threshold).astype(int)
    tn_opt, fp_opt, fn_opt, tp_opt = confusion_matrix(y_true, preds_opt, labels=[0,1]).ravel()
    f1_opt = f1_score(y_true, preds_opt, zero_division=0)
    prec_opt = precision_score(y_true, preds_opt, zero_division=0)
    rec_opt = recall_score(y_true, preds_opt, zero_division=0)
    acc_opt = accuracy_score(y_true, preds_opt)
    
    # AUPRC
    auprc = average_precision_score(y_true, y_prob)

    # --- 3. Print Full Report ---
    print(f"\n--- FINAL {set_name.upper()} RESULTS ({team.upper()}) ---")
    print(f"  {team.upper()} Team Weighted Log Loss: {weighted_loss:.4f}")
    print(f"  {team.upper()} Team AUPRC: {auprc:.4f}")
    print(f"\n-- Default @ 0.5 Threshold --")
    print(f"  F1: {f1_def:.4f} | P: {prec_def:.4f} | R: {rec_def:.4f} | Acc: {acc_def:.4f}")
    print(f"    -> TP: {tp_def} | TN: {tn_def} | FP: {fp_def} | FN: {fn_def}")
    print(f"\n-- Custom @ {threshold} Threshold --")
    print(f"  F1: {f1_opt:.4f} | P: {prec_opt:.4f} | R: {rec_opt:.4f} | Acc: {acc_opt:.4f}")
    print(f"    -> TP: {tp_opt} | TN: {tn_opt} | FP: {fp_opt} | FN: {fn_opt}")

    # --- 4. Generate Figures (Only for Test Set) ---
    if set_name.lower() == 'test':
        print("\n--- Generating Figures for TEST set ---")
        model_type_str = f'lstm_h{model.lstm.hidden_size}_l{model.lstm.num_layers}'
        
        plot_and_save_distribution(y_true, y_prob, threshold, model_type_str, team, "Test", output_dir)
        
        plot_and_save_confusion_matrix(y_true, preds_def, 0.5, model_type_str, team, "Test", output_dir)
        plot_and_save_confusion_matrix(y_true, preds_opt, threshold, model_type_str, team, "Test", output_dir)
        
        metrics_def_tuple = (prec_def, rec_def, f1_def)
        metrics_opt_tuple = (prec_opt, rec_opt, f1_opt)
        plot_and_save_metrics_table(metrics_def_tuple, metrics_opt_tuple, threshold, model_type_str, team, "Test", output_dir)

# ====================== MAIN ANALYSIS ======================
def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Full Analysis for LSTM Model")
    
    parser.add_argument('--model-path', type=str, required=True, help='Path to the saved LSTM model checkpoint (.pth file).')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to the parent directory of train/val/test splits.')
    parser.add_argument('--team', type=str, required=True, choices=['orange', 'blue'], help='Which team to analyze.')
    parser.add_argument('--threshold', type=float, required=True, help='The custom threshold to use for analysis.')
    parser.add_argument('--output-dir', type=str, default='./analysis_results_lstm', help='Directory to save all output figures.')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size for evaluation.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA evaluation.')
    
    args = parser.parse_args()
    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")
    
    # Create output directory
    output_dir_team = os.path.join(args.output_dir, f"lstm_{args.team}")
    os.makedirs(output_dir_team, exist_ok=True)
    print(f"--- Saving outputs to: {output_dir_team} ---")

    # --- 1. Calculate Class Weights (Needed for Loss) ---
    print("\n--- Calculating class weights from training data ---")
    train_dir = os.path.join(args.data_dir, 'train')
    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.csv')]
    if not train_files:
        print(f"CRITICAL ERROR: No training files found in {train_dir}. Cannot calculate weights."); return
    
    pos_weight_orange, pos_weight_blue = calculate_class_weights(train_files)

    # --- 2. Load Data ---
    print(f"\n--- Loading VAL and TEST data ---")
    val_dir = os.path.join(args.data_dir, 'val'); test_dir = os.path.join(args.data_dir, 'test')
    val_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.csv')]
    test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.csv')]
    
    val_dataset = SequentialLazyDataset(val_files, sequence_length=SEQUENCE_LENGTH)
    test_dataset = SequentialLazyDataset(test_files, sequence_length=SEQUENCE_LENGTH)
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True, collate_fn=collate_fn_master)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True, collate_fn=collate_fn_master)

    # --- 3. Load Model ---
    print(f"\n--- Loading model from {args.model_path} ---")
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        model_args = checkpoint.get('args', {})
        
        # Load hyperparameters from checkpoint
        hidden_dim = model_args.get('hidden_dim', 128) # default 128
        num_layers = model_args.get('num_layers', 1) # default 1
        dropout = model_args.get('dropout', 0.3) # default 0.3
            
        print(f"Initializing LSTM with: HiddenDim={hidden_dim}, NumLayers={num_layers}, Dropout={dropout}")
        model = BaselineLSTM(
            input_dim=TOTAL_FLAT_FEATURES, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers,
            dropout=dropout
        ).to(device)
        
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load the model. Ensure path and model class match. Error: {e}"); return

    # --- 4. Run Analysis on VALIDATION Set ---
    run_full_analysis(model, val_loader, device, 
                      pos_weight_orange, pos_weight_blue, 
                      args.team, args.threshold, 
                      "Validation", output_dir_team)
    
    # --- 5. Run Analysis on TEST Set ---
    run_full_analysis(model, test_loader, device, 
                      pos_weight_orange, pos_weight_blue, 
                      args.team, args.threshold, 
                      "Test", output_dir_team)

    end_time = time.time()
    total_seconds = end_time - start_time
    print(f"\n--- Analysis Complete ---")
    print(f"\n--- Total Run Time: {total_seconds // 3600:.0f}h {(total_seconds % 3600) // 60:.0f}m {total_seconds % 60:.2f}s ---")


if __name__ == '__main__':
    main()