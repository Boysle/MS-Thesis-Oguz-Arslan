import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time  # ##### NEW/FIXED #####

# ##### NEW/FIXED #####: Added accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, precision_recall_curve, average_precision_score, accuracy_score
from tqdm import tqdm
import wandb
import linecache

# ====================== CONFIGURATION & CONSTANTS ======================
NUM_PLAYERS = 6
PLAYER_FEATURES = 13
GLOBAL_FEATURES = 14 
TOTAL_FLAT_FEATURES = (NUM_PLAYERS * PLAYER_FEATURES) + GLOBAL_FEATURES

# Normalization bounds
POS_MIN_X, POS_MAX_X = -4096, 4096; POS_MIN_Y, POS_MAX_Y = -6000, 6000; POS_MIN_Z, POS_MAX_Z = 0, 2044
VEL_MIN, VEL_MAX = -2300, 2300; BOOST_MIN, BOOST_MAX = 0, 100; BALL_VEL_MIN, BALL_VEL_MAX = -6000, 6000
BOOST_PAD_MIN, BOOST_PAD_MAX = 0, 10; DIST_MIN, DIST_MAX = 0, (8192**2 + 10240**2 + 2044**2)**(1/2)

def normalize(val, min_val, max_val): return (val - min_val) / (max_val - min_val + 1e-8)

def parse_args():
    parser = argparse.ArgumentParser(description="Rocket League Baseline MLP Training")
    parser.add_argument('--data-dir', type=str, default="F:\\Raw RL Esports Replays\\Day 3 Swiss Stage\\Round 1\\split_dataset", help='Parent directory of splits.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size.')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA.')
    parser.add_argument('--wandb-project', type=str, default="rl-goal-prediction-baseline", help="W&B project name.")
    parser.add_argument('--run-name', type=str, default=None, help="Custom name for the W&B run.")
    parser.add_argument('--resume', action='store_true', help='Resume training from a checkpoint.')
    parser.add_argument('--checkpoint-path', type=str, default='./baseline_checkpoint.pth', help='Path for checkpoint.')
    parser.add_argument('--checkpoint-every', type=int, default=5, help='Save checkpoint every N epochs.')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of workers for DataLoader.')
    return parser.parse_args()

# ====================== DATASET CLASS (Fast & Robust) ======================
class RobustLazyDataset(Dataset):
    def __init__(self, list_of_csv_paths):
        self.csv_paths = list_of_csv_paths; self.file_info = []; self.cumulative_rows = [0]; self.header = None; total_rows = 0
        desc = f"Indexing {os.path.basename(os.path.dirname(list_of_csv_paths[0]))} files"
        for path in tqdm(self.csv_paths, desc=desc):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    if self.header is None: self.header = f.readline().strip().split(',')
                    num_lines = sum(1 for _ in f)
                if num_lines > 0: self.file_info.append({'path': path, 'rows': num_lines}); total_rows += num_lines; self.cumulative_rows.append(total_rows)
            except Exception as e: print(f"\nWarning: Could not process file {path}. Skipping. Error: {e}")
        self.length = total_rows
        print(f"\n--- Indexing complete. Total samples: {self.length} ---")
    def __len__(self): return self.length
    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length: raise IndexError("Index out of range")
        file_index = np.searchsorted(self.cumulative_rows, idx, side='right') - 1; file_path = self.file_info[file_index]['path']; local_idx = idx - self.cumulative_rows[file_index]
        try:
            line = linecache.getline(file_path, local_idx + 2)
            if not line: return None
            row = dict(zip(self.header, line.strip().split(',')))
            player_features = [item for i in range(NUM_PLAYERS) for item in [normalize(float(row[f'p{i}_pos_x']), POS_MIN_X, POS_MAX_X), normalize(float(row[f'p{i}_pos_y']), POS_MIN_Y, POS_MAX_Y), normalize(float(row[f'p{i}_pos_z']), POS_MIN_Z, POS_MAX_Z), normalize(float(row[f'p{i}_vel_x']), VEL_MIN, VEL_MAX), normalize(float(row[f'p{i}_vel_y']), VEL_MIN, VEL_MAX), normalize(float(row[f'p{i}_vel_z']), VEL_MIN, VEL_MAX), float(row[f'p{i}_forward_x']), float(row[f'p{i}_forward_y']), float(row[f'p{i}_forward_z']), normalize(float(row[f'p{i}_boost_amount']), BOOST_MIN, BOOST_MAX), float(row[f'p{i}_team']), float(row[f'p{i}_alive']), normalize(float(row[f'p{i}_dist_to_ball']), DIST_MIN, DIST_MAX)]]
            global_features = [normalize(float(row['ball_pos_x']), POS_MIN_X, POS_MAX_X), normalize(float(row['ball_pos_y']), POS_MIN_Y, POS_MAX_Y), normalize(float(row['ball_pos_z']), POS_MIN_Z, POS_MAX_Z), normalize(float(row['ball_vel_x']), BALL_VEL_MIN, BALL_VEL_MAX), normalize(float(row['ball_vel_y']), BALL_VEL_MIN, BALL_VEL_MAX), normalize(float(row['ball_vel_z']), BALL_VEL_MIN, BALL_VEL_MAX), normalize(float(row['boost_pad_0_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX), normalize(float(row['boost_pad_1_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX), normalize(float(row['boost_pad_2_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX), normalize(float(row['boost_pad_3_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX), normalize(float(row['boost_pad_4_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX), normalize(float(row['boost_pad_5_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX), float(row['ball_hit_team_num']), normalize(min(float(row['seconds_remaining']), 300.0), 0, 300)]
            features = torch.tensor(player_features + global_features, dtype=torch.float32)
            orange_y = torch.tensor([float(row['team_1_goal_in_event_window'])], dtype=torch.float32)
            blue_y = torch.tensor([float(row['team_0_goal_in_event_window'])], dtype=torch.float32)
            return features, orange_y, blue_y
        except (ValueError, KeyError, IndexError): return None

def collate_fn_skip_none(batch):
    batch = [item for item in batch if item is not None];
    if not batch: return None
    return torch.utils.data.dataloader.default_collate(batch)

# ====================== HELPER FUNCTION ======================

# Added this helper function
def find_optimal_threshold(y_true, y_pred_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    # Add epsilon to prevent division by zero
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    # Exclude the last value which doesn't have a corresponding threshold
    best_f1_idx = np.argmax(f1_scores[:-1]) 
    return thresholds[best_f1_idx], f1_scores[best_f1_idx]

# ====================== BASELINE MLP MODEL ======================
class BaselineMLP(nn.Module):
    def __init__(self, input_dim=TOTAL_FLAT_FEATURES):
        super(BaselineMLP, self).__init__()
        
        self.body = nn.Sequential(
            nn.Linear(input_dim, 256), 
            nn.BatchNorm1d(256), 
            nn.ReLU(), 
            nn.Dropout(0.5), 
            nn.Linear(256, 128), 
            nn.BatchNorm1d(128), 
            nn.ReLU(), 
            nn.Dropout(0.5)
        )
        
        # Output raw logits, not probabilities. Removed nn.Sigmoid()
        self.orange_head = nn.Sequential(nn.Linear(128, 1))
        self.blue_head = nn.Sequential(nn.Linear(128, 1))

    def forward(self, x): 
        x_p = self.body(x)
        return self.orange_head(x_p), self.blue_head(x_p)

# ====================== MAIN EXECUTION ======================
def main():
    ##### NEW/FIXED #####
    start_time = time.time()

    args = parse_args(); use_cuda = not args.no_cuda and torch.cuda.is_available(); device = torch.device("cuda" if use_cuda else "cpu"); print(f"--- Using device: {device} ---")

    # Changed from best_val_f1 to best_val_loss
    start_epoch, best_val_loss, wandb_run_id = 0, np.inf, None
    
    if args.resume:
        if os.path.exists(args.checkpoint_path):
            print(f"--- Found checkpoint. Attempting to resume: {args.checkpoint_path} ---")
            checkpoint = torch.load(args.checkpoint_path, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            
            # Load best_val_loss, fallback to infinity
            best_val_loss = checkpoint.get('best_val_loss', np.inf) 
            wandb_run_id = checkpoint.get('wandb_run_id')
            
            print(f"--- Checkpoint loaded. Resuming from epoch {start_epoch}. Best Val Loss: {best_val_loss:.4f} ---")
            if wandb_run_id: print(f"--- Resuming W&B run: {wandb_run_id} ---")
        else:
            print(f"--- WARNING: --resume flag was set, but no checkpoint found. Starting fresh. ---")

    try:
        wandb.init(project=args.wandb_project, name=args.run_name, config=args, id=wandb_run_id, resume="allow")
        print("--- Weights & Biases successfully initialized ---")
    except Exception as e:
        print(f"--- Could not initialize W&B: {e}. Running without logging. ---"); wandb.run = None

    print("\n--- Initializing Data Loaders (Robust, Fast Lazy Loading) ---")
    train_dir = os.path.join(args.data_dir, 'train'); val_dir = os.path.join(args.data_dir, 'val')
    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.csv')]
    val_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.csv')]
    train_dataset = RobustLazyDataset(train_files); val_dataset = RobustLazyDataset(val_files)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_skip_none)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_skip_none)

    # Calculate class weights for imbalance
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

    model = BaselineMLP().to(device); optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Use BCEWithLogitsLoss for numerical stability and to apply class weights
    criterion_orange = nn.BCEWithLogitsLoss(pos_weight=pos_weight_orange)
    criterion_blue = nn.BCEWithLogitsLoss(pos_weight=pos_weight_blue)

    if args.resume and 'checkpoint' in locals(): 
        model.load_state_dict(checkpoint['model_state']); 
        if 'optimizer_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state'])

    print(f"\n--- Starting Training from epoch {start_epoch + 1} to {args.epochs} ---")
    for epoch in range(start_epoch, args.epochs):
        model.train(); total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [TRAIN]")
        for batch in pbar:
            if batch is None: continue
            features, orange_labels, blue_labels = batch; features, orange_labels, blue_labels = features.to(device), orange_labels.to(device), blue_labels.to(device)
            
            optimizer.zero_grad()
            orange_logits, blue_logits = model(features) # Model outputs logits
            
            loss = criterion_orange(orange_logits, orange_labels) + criterion_blue(blue_logits, blue_labels)
            
            loss.backward(); optimizer.step(); total_train_loss += loss.item()

        model.eval(); total_val_loss = 0
        val_probs_o, val_labels_o, val_probs_b, val_labels_b = [], [], [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [VAL]"):
                if batch is None: continue
                features, orange_labels, blue_labels = batch; features, orange_labels, blue_labels = features.to(device), orange_labels.to(device), blue_labels.to(device)
                
                orange_logits, blue_logits = model(features) # Get logits
                
                total_val_loss += (criterion_orange(orange_logits, orange_labels) + criterion_blue(blue_logits, blue_labels)).item()
                
                # Apply sigmoid manually to get probs for metric calculation
                orange_pred_prob = torch.sigmoid(orange_logits)
                blue_pred_prob = torch.sigmoid(blue_logits)
                
                val_probs_o.extend(orange_pred_prob.cpu().numpy().flatten())
                val_labels_o.extend(orange_labels.cpu().numpy().flatten())
                val_probs_b.extend(blue_pred_prob.cpu().numpy().flatten())
                val_labels_b.extend(blue_labels.cpu().numpy().flatten())
                
        avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        # --- Calculate all validation metrics ---
        np_val_labels_o = np.array(val_labels_o)
        np_val_probs_o = np.array(val_probs_o)
        np_val_labels_b = np.array(val_labels_b)
        np_val_probs_b = np.array(val_probs_b)
        
        # Calculate @ 0.5 threshold
        val_preds_o_binary = (np_val_probs_o > 0.5).astype(int)
        val_preds_b_binary = (np_val_probs_b > 0.5).astype(int)
        
        val_f1_o = f1_score(np_val_labels_o, val_preds_o_binary, zero_division=0)
        val_f1_b = f1_score(np_val_labels_b, val_preds_b_binary, zero_division=0)
        avg_val_f1_at_05 = (val_f1_o + val_f1_b) / 2
        
        val_prec_o = precision_score(np_val_labels_o, val_preds_o_binary, zero_division=0)
        val_recall_o = recall_score(np_val_labels_o, val_preds_o_binary, zero_division=0)
        val_prec_b = precision_score(np_val_labels_b, val_preds_b_binary, zero_division=0)
        val_recall_b = recall_score(np_val_labels_b, val_preds_b_binary, zero_division=0)

        # Calculate AUPRC (threshold-independent)
        val_auprc_o = average_precision_score(np_val_labels_o, np_val_probs_o)
        val_auprc_b = average_precision_score(np_val_labels_b, np_val_probs_b)
        avg_val_auprc = (val_auprc_o + val_auprc_b) / 2

        print(f"\nEpoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Avg Val F1@0.5: {avg_val_f1_at_05:.4f} | Avg Val AUPRC: {avg_val_auprc:.4f}")
        
        if wandb.run:
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
            })

        current_wandb_id = wandb.run.id if wandb.run else None
        
        # This is the main checkpointing fix.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(os.path.dirname(args.checkpoint_path), 'best_mlp_model.pth') # Use specific name
            print(f"  *** New best model found (Val Loss: {best_val_loss:.4f} at epoch {epoch+1}). Saving 'best' checkpoint. ***")
            torch.save({
                'epoch': epoch, 
                'model_state': model.state_dict(), 
                'best_val_loss': best_val_loss, # Save the loss
                'args': vars(args), # Save args dict
                'wandb_run_id': current_wandb_id
            }, best_model_path)
        
        if (epoch + 1) % args.checkpoint_every == 0 or (epoch + 1) == args.epochs:
            print(f"--- Saving periodic checkpoint at epoch {epoch + 1}. ---")
            torch.save({
                'epoch': epoch, 
                'model_state': model.state_dict(), 
                'optimizer_state': optimizer.state_dict(), 
                'best_val_loss': best_val_loss, 
                'args': vars(args), # Save args dict
                'wandb_run_id': current_wandb_id
            }, args.checkpoint_path)

    # Finish the training-loop W&B run
    if wandb.run and wandb.run.id == wandb_run_id and wandb.run.resumed is False:
        wandb.finish()

    print("\n--- Training Complete ---")
    
    ##### NEW/FIXED #####
    # ================= FINAL VALIDATION & TEST EVALUATION ============================
    print("\n--- Starting Final Evaluation ---")
    best_model_path = os.path.join(os.path.dirname(args.checkpoint_path), 'best_mlp_model.pth') 

    if not os.path.exists(best_model_path):
        print("--- No 'best_mlp_model.pth' checkpoint found. Skipping final test evaluation. ---")
    else:
        print(f"--- Loading best model from: {best_model_path} for final evaluation ---")
        checkpoint = torch.load(best_model_path, map_location=device)
        # Re-initialize model to ensure correct architecture
        model = BaselineMLP().to(device) 
        model.load_state_dict(checkpoint['model_state'])
        model.eval()

        # --- Step 1: Find Optimal Threshold on the FULL Validation Set ---
        print("\n--- Determining optimal thresholds on the validation set... ---")
        all_val_oprobs, all_val_olabels, all_val_bprobs, all_val_blabels = [], [], [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="[FINAL VAL]"):
                if batch is None: continue
                features, orange_labels, blue_labels = batch; features, orange_labels, blue_labels = features.to(device), orange_labels.to(device), blue_labels.to(device)
                orange_logits, blue_logits = model(features)
                all_val_oprobs.extend(torch.sigmoid(orange_logits).cpu().numpy().flatten())
                all_val_olabels.extend(orange_labels.cpu().numpy().flatten())
                all_val_bprobs.extend(torch.sigmoid(blue_logits).cpu().numpy().flatten())
                all_val_blabels.extend(blue_labels.cpu().numpy().flatten())
        
        optimal_threshold_orange, _ = find_optimal_threshold(all_val_olabels, all_val_oprobs)
        optimal_threshold_blue, _ = find_optimal_threshold(all_val_blabels, all_val_bprobs)

        print(f"  Optimal Threshold (Orange): {optimal_threshold_orange:.4f}")
        print(f"  Optimal Threshold (Blue):   {optimal_threshold_blue:.4f}")

        # --- Step 2: Run Evaluation on the Test Set ---
        print("\n--- Running final evaluation on the test set... ---")
        test_dir = os.path.join(args.data_dir, 'test'); test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.csv')]
        
        if not test_files:
            print("--- No test files found. Skipping. ---")
            test_dataset = None
        else:
            test_dataset = RobustLazyDataset(test_files)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_skip_none)
            print(f"Total test samples: {len(test_dataset)}")
            
            all_test_oprobs, all_test_olabels, all_test_bprobs, all_test_blabels = [], [], [], []
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="[FINAL TEST]"):
                    if batch is None: continue
                    features, orange_labels, blue_labels = batch; features, orange_labels, blue_labels = features.to(device), orange_labels.to(device), blue_labels.to(device)
                    orange_logits, blue_logits = model(features)
                    all_test_oprobs.extend(torch.sigmoid(orange_logits).cpu().numpy().flatten())
                    all_test_olabels.extend(orange_labels.cpu().numpy().flatten())
                    all_test_bprobs.extend(torch.sigmoid(blue_logits).cpu().numpy().flatten())
                    all_test_blabels.extend(blue_labels.cpu().numpy().flatten())

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
                wandb.summary["optimized_test_accuracy_orange"] = acc_opt_o
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
                
    ##### NEW/FIXED #####
    # Final time printout
    end_time_final = time.time()
    total_seconds_final = end_time_final - start_time
    print(f"\n--- Total Run Time: {total_seconds_final // 3600:.0f}h {(total_seconds_final % 3600) // 60:.0f}m {total_seconds_final % 60:.2f}s ---")


if __name__ == '__main__':
    main()