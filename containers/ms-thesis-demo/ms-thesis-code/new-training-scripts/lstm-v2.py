import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import linecache
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score, precision_recall_curve, 
    confusion_matrix, accuracy_score, average_precision_score, log_loss
)
import wandb

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

# ====================== DATASET CLASS (Sequential w/ Outlier Handling) ======================
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
        outlier_count_replay, outlier_count_score = 0, 0
        
        anchor_row, anchor_file_idx = self._get_row(current_idx)
        if anchor_row is None:
            return None
        
        anchor_replay_id = anchor_row['replay_id']
        anchor_score_diff = anchor_row['score_difference']
        
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

        return x_tensor, y_orange, y_blue, outlier_count_replay, outlier_count_score

# ====================== LSTM MODEL (with Dropout) ======================
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
    x_tensors, y_oranges, y_blues, replay_outliers, score_outliers = zip(*batch)
    x_batch = torch.stack(x_tensors)
    y_o_batch = torch.stack(y_oranges)
    y_b_batch = torch.stack(y_blues)
    total_replay_outliers = sum(replay_outliers)
    total_score_outliers = sum(score_outliers)
    return x_batch, y_o_batch, y_b_batch, total_replay_outliers, total_score_outliers

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
    total_replay_outliers, total_score_outliers = 0, 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Getting predictions and loss"):
            if batch is None: continue
            x_batch, y_o_batch, y_b_batch, replay_outliers, score_outliers = batch
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

    num_samples = len(all_orange_labels)
    avg_loss_o = total_loss_o / num_samples if num_samples > 0 else 0
    avg_loss_b = total_loss_b / num_samples if num_samples > 0 else 0
            
    return (np.array(all_orange_labels), np.array(all_blue_labels), 
            np.array(all_orange_probs), np.array(all_blue_probs),
            avg_loss_o, avg_loss_b,
            total_replay_outliers, total_score_outliers)

# ====================== ARGUMENT PARSER ======================
def parse_args():
    parser = argparse.ArgumentParser(description="Rocket League Baseline LSTM Training (v4 w/ Early Stopping)")
    parser.add_argument('--data-dir', type=str, default="E:\\...\\split_dataset", help='Parent directory of splits.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    
    ##### NEW/MODIFIED #####
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size (default: 128).')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate (default: 0.0001).')
    parser.add_argument('--hidden-dim', type=int, default=128, help='LSTM hidden dimension.')
    parser.add_argument('--num-layers', type=int, default=1, help='Number of LSTM layers.')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout probability (default: 0.3).')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Adam weight decay (L2 reg) (default: 1e-5).')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience in epochs (default: 5).')
    ##### END NEW/MODIFIED #####
    
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA.')
    parser.add_argument('--wandb-project', type=str, default="rl-lstm-baseline", help="W&B project name.")
    parser.add_argument('--run-name', type=str, default=None, help="Custom name for the W&B run.")
    parser.add_argument('--resume', action='store_true', help='Resume training from a checkpoint.')
    parser.add_argument('--checkpoint-path', type=str, default='./lstm_checkpoint.pth', help='Path for checkpoint.')
    parser.add_argument('--checkpoint-every', type=int, default=5, help='Save checkpoint every N epochs.')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of workers for DataLoader.')
    return parser.parse_args()

# ====================== MAIN EXECUTION ======================
def main():
    start_time = time.time()
    args = parse_args()
    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"--- Using device: {device} ---")

    start_epoch, best_val_loss, wandb_run_id = 0, np.inf, None
    
    ##### NEW/MODIFIED #####
    # Initialize epochs_no_improve for early stopping
    epochs_no_improve = 0
    
    if args.resume and os.path.exists(args.checkpoint_path):
        print(f"--- Resuming from checkpoint: {args.checkpoint_path} ---")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', np.inf) 
        wandb_run_id = checkpoint.get('wandb_run_id')
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0) # Load patience counter
        print(f"--- Resuming from epoch {start_epoch}. Best Val Loss: {best_val_loss:.4f} ---")
    
    try:
        wandb.init(project=args.wandb_project, name=args.run_name, config=args, id=wandb_run_id, resume="allow")
    except Exception as e:
        print(f"--- Could not initialize W&B: {e}. Running without logging. ---"); wandb.run = None

    # --- 1. Load Data & Weights ---
    train_dir = os.path.join(args.data_dir, 'train'); val_dir = os.path.join(args.data_dir, 'val')
    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.csv')]
    val_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.csv')]
    
    pos_weight_orange, pos_weight_blue = calculate_class_weights(train_files)
    
    train_dataset = SequentialLazyDataset(train_files, sequence_length=SEQUENCE_LENGTH)
    val_dataset = SequentialLazyDataset(val_files, sequence_length=SEQUENCE_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_master)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_master)

    # --- 2. Initialize Model & Criteria ---
    model = BaselineLSTM(
        input_dim=TOTAL_FLAT_FEATURES, 
        hidden_dim=args.hidden_dim, 
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    
    # --- LR Scheduler Removed ---
    
    criterion_orange = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_orange]).to(device))
    criterion_blue = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_blue]).to(device))
    
    if args.resume and 'checkpoint' in locals(): 
        model.load_state_dict(checkpoint['model_state'])
        if 'optimizer_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state'])

    # --- 3. Training Loop ---
    print(f"\n--- Starting Training from epoch {start_epoch + 1} to {args.epochs} ---")
    for epoch in range(start_epoch, args.epochs):
        model.train(); total_train_loss = 0
        epoch_replay_outliers, epoch_score_outliers = 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [TRAIN]")
        for batch in pbar:
            if batch is None: continue
            x_batch, y_o_batch, y_b_batch, replay_outliers, score_outliers = batch
            x_batch, y_o_batch, y_b_batch = x_batch.to(device), y_o_batch.to(device), y_b_batch.to(device)
            
            epoch_replay_outliers += replay_outliers
            epoch_score_outliers += score_outliers
            
            optimizer.zero_grad()
            orange_logits, blue_logits = model(x_batch)
            loss = criterion_orange(orange_logits, y_o_batch) + criterion_blue(blue_logits, y_b_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0

        # --- Validation Loop ---
        (val_labels_o, val_labels_b, val_probs_o, val_probs_b, 
         val_loss_o, val_loss_b, val_replay_outliers, val_score_outliers) = get_predictions_and_loss_sequential(
             model, val_loader, device, criterion_orange, criterion_blue)
        
        avg_val_loss = val_loss_o + val_loss_b # Total loss
        
        # (Calculate other validation metrics for logging)
        val_f1_o = f1_score(val_labels_o, val_probs_o > 0.5, zero_division=0)
        val_f1_b = f1_score(val_labels_b, val_probs_b > 0.5, zero_division=0)
        avg_val_f1_at_05 = (val_f1_o + val_f1_b) / 2
        val_auprc_o = average_precision_score(val_labels_o, val_probs_o)
        val_auprc_b = average_precision_score(val_labels_b, val_probs_b)
        avg_val_auprc = (val_auprc_o + val_auprc_b) / 2
        
        print(f"\nEpoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Avg Val F1@0.5: {avg_val_f1_at_05:.4f} | Avg Val AUPRC: {avg_val_auprc:.4f}")
        print(f"  Train Outliers: Replay={epoch_replay_outliers}, Score={epoch_score_outliers}")
        print(f"  Val Outliers:   Replay={val_replay_outliers}, Score={val_score_outliers}")

        if wandb.run:
            wandb.log({
                "epoch": epoch + 1, "train/loss": avg_train_loss, "val/loss": avg_val_loss,
                "val/f1_orange_at_0.5": val_f1_o, "val/f1_blue_at_0.5": val_f1_b, "val/avg_f1_at_0.5": avg_val_f1_at_05,
                "val/auprc_orange": val_auprc_o, "val/auprc_blue": val_auprc_b, "val/avg_auprc": avg_val_auprc,
                "outliers/train_replay": epoch_replay_outliers, "outliers/train_score": epoch_score_outliers,
                "outliers/val_replay": val_replay_outliers, "outliers/val_score": val_score_outliers,
                "learning_rate": optimizer.param_groups[0]['lr']
            })

        current_wandb_id = wandb.run.id if wandb.run else None
        
        # Checkpointing and Early Stopping Logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0 # Reset patience
            best_model_path = os.path.join(os.path.dirname(args.checkpoint_path), 'best_lstm_model.pth')
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
            break # Exit the training loop

    if wandb.run and wandb.run.id == wandb_run_id and wandb.run.resumed is False:
        wandb.finish()
    print("\n--- Script Finished Training ---")

    # ================= FINAL VALIDATION & TEST EVALUATION ============================
    # (This section remains identical to the previous script)
    print("\n--- Starting Final Evaluation ---")
    best_model_path = os.path.join(os.path.dirname(args.checkpoint_path), 'best_lstm_model.pth') 

    if not os.path.exists(best_model_path):
        print("--- No 'best_lstm_model.pth' checkpoint found. Skipping final test evaluation. ---")
    else:
        print(f"--- Loading best model from: {best_model_path} for final evaluation ---")
        checkpoint = torch.load(best_model_path, map_location=device)
        
        cp_args = checkpoint.get('args', {})
        hidden_dim = cp_args.get('hidden_dim', args.hidden_dim)
        num_layers = cp_args.get('num_layers', args.num_layers)
        dropout = cp_args.get('dropout', args.dropout)
        
        model = BaselineLSTM(
            input_dim=TOTAL_FLAT_FEATURES, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers,
            dropout=dropout
        ).to(device)
        model.load_state_dict(checkpoint['model_state'])
        
        criterion_o = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_orange]).to(device))
        criterion_b = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_blue]).to(device))

        # --- Step 1: Find Optimal Threshold on the FULL Validation Set ---
        print("\n--- Determining optimal thresholds on the validation set... ---")
        (val_labels_o, val_labels_b, val_probs_o, val_probs_b, 
         val_loss_o, val_loss_b, v_rep, v_sco) = get_predictions_and_loss_sequential(
            model, val_loader, device, criterion_o, criterion_b)
        
        optimal_threshold_orange, _ = find_optimal_threshold(val_labels_o, val_probs_o)
        optimal_threshold_blue, _ = find_optimal_threshold(val_labels_b, val_probs_b)
        print(f"  Optimal Threshold (Orange): {optimal_threshold_orange:.4f}")
        print(f"  Optimal Threshold (Blue):   {optimal_threshold_blue:.4f}")
        print(f"  Val Total Loss: {val_loss_o + val_loss_b:.4f} (O:{val_loss_o:.4f}, B:{val_loss_b:.4f})")
        print(f"  Val Outliers Found: Replay={v_rep}, Score={v_sco}")


        # --- Step 2: Run Evaluation on the Test Set ---
        print("\n--- Running final evaluation on the test set... ---")
        test_dir = os.path.join(args.data_dir, 'test')
        test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.csv')]
        
        if not test_files:
            print("--- No test files found. Skipping. ---")
        else:
            test_dataset = SequentialLazyDataset(test_files, sequence_length=SEQUENCE_LENGTH)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_master)
            
            (y_true_o, y_true_b, y_prob_o, y_prob_b, 
             test_loss_o, test_loss_b, t_rep, t_sco) = get_predictions_and_loss_sequential(
                model, test_loader, device, criterion_o, criterion_b)
            print(f"  Test Outliers Found: Replay={t_rep}, Score={t_sco}")

            # --- Step 3: Calculate Metrics for Test Set (Optimized & Default) ---
            # (Full metric calculation block)
            
            # --- Default @ 0.5 ---
            preds_def_o = (y_prob_o > 0.5).astype(int); preds_def_b = (y_prob_b > 0.5).astype(int)
            tn_def_o, fp_def_o, fn_def_o, tp_def_o = confusion_matrix(y_true_o, preds_def_o, labels=[0,1]).ravel()
            f1_def_o = f1_score(y_true_o, preds_def_o, zero_division=0); prec_def_o = precision_score(y_true_o, preds_def_o, zero_division=0); rec_def_o = recall_score(y_true_o, preds_def_o, zero_division=0); acc_def_o = accuracy_score(y_true_o, preds_def_o)
            tn_def_b, fp_def_b, fn_def_b, tp_def_b = confusion_matrix(y_true_b, preds_def_b, labels=[0,1]).ravel()
            f1_def_b = f1_score(y_true_b, preds_def_b, zero_division=0); prec_def_b = precision_score(y_true_b, preds_def_b, zero_division=0); rec_def_b = recall_score(y_true_b, preds_def_b, zero_division=0); acc_def_b = accuracy_score(y_true_b, preds_def_b)

            # --- Optimized ---
            preds_opt_o = (y_prob_o > optimal_threshold_orange).astype(int); preds_opt_b = (y_prob_b > optimal_threshold_blue).astype(int)
            tn_opt_o, fp_opt_o, fn_opt_o, tp_opt_o = confusion_matrix(y_true_o, preds_opt_o, labels=[0,1]).ravel()
            f1_opt_o = f1_score(y_true_o, preds_opt_o, zero_division=0); prec_opt_o = precision_score(y_true_o, preds_opt_o, zero_division=0); rec_opt_o = recall_score(y_true_o, preds_opt_o, zero_division=0); acc_opt_o = accuracy_score(y_true_o, preds_opt_o)
            tn_opt_b, fp_opt_b, fn_opt_b, tp_opt_b = confusion_matrix(y_true_b, preds_opt_b, labels=[0,1]).ravel()
            f1_opt_b = f1_score(y_true_b, preds_opt_b, zero_division=0); prec_opt_b = precision_score(y_true_b, preds_opt_b, zero_division=0); rec_opt_b = recall_score(y_true_b, preds_opt_b, zero_division=0); acc_opt_b = accuracy_score(y_true_b, preds_opt_b)

            # --- AUPRC (Threshold-Independent) ---
            auprc_o = average_precision_score(y_true_o, y_prob_o); auprc_b = average_precision_score(y_true_b, y_prob_b)

            # --- Step 3b: New Print Block ---
            print("\n--- FINAL TEST RESULTS ---")
            print(f"  Total Weighted Log Loss: {test_loss_o + test_loss_b:.4f} (O: {test_loss_o:.4f}, B: {test_loss_b:.4f})")
            print("\n-- Default @ 0.5 Threshold --")
            print(f"  Orange Team: F1: {f1_def_o:.4f} | P: {prec_def_o:.4f} | R: {rec_def_o:.4f} | Acc: {acc_def_o:.4f}"); print(f"    -> TP: {tp_def_o} | TN: {tn_def_o} | FP: {fp_def_o} | FN: {fn_def_o}")
            print(f"  Blue Team:   F1: {f1_def_b:.4f} | P: {prec_def_b:.4f} | R: {rec_def_b:.4f} | Acc: {acc_def_b:.4f}"); print(f"    -> TP: {tp_def_b} | TN: {tn_def_b} | FP: {fp_def_b} | FN: {fn_def_b}")
            print("\n-- Optimized Threshold --")
            print(f"  Orange Team (@ {optimal_threshold_orange:.3f}): F1: {f1_opt_o:.4f} | P: {prec_opt_o:.4f} | R: {rec_opt_o:.4f} | Acc: {acc_opt_o:.4f}"); print(f"    -> TP: {tp_opt_o} | TN: {tn_opt_o} | FP: {fp_opt_o} | FN: {fn_opt_o}")
            print(f"  Blue Team   (@ {optimal_threshold_blue:.3f}): F1: {f1_opt_b:.4f} | P: {prec_opt_b:.4f} | R: {rec_opt_b:.4f} | Acc: {acc_opt_b:.4f}"); print(f"    -> TP: {tp_opt_b} | TN: {tn_opt_b} | FP: {fp_opt_b} | FN: {fn_opt_b}")
            print("\n-- Threshold-Independent --")
            print(f"  AUPRC (Orange): {auprc_o:.4f}"); print(f"  AUPRC (Blue):   {auprc_b:.4f}")

            # --- Step 4: Log Final Summary to W&B ---
            try:
                if wandb.run is None: 
                    wandb.init(project=args.wandb_project, id=wandb_run_id, resume="must")
                
                print("\n--- Logging final summary to W&B ---")
                wandb.summary["best_epoch"] = checkpoint.get('epoch', 0) + 1
                wandb.summary["best_val_loss_at_save"] = checkpoint.get('best_val_loss', 0.0)
                
                # (Full logging of default and optimized metrics...)
                
                # Log final outlier counts
                wandb.summary["total_val_replay_outliers"] = v_rep
                wandb.summary["total_val_score_outliers"] = v_sco
                wandb.summary["total_test_replay_outliers"] = t_rep
                wandb.summary["total_test_score_outliers"] = t_sco

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