import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
import wandb
import linecache  # Import the efficient line-reading module

# ====================== CONFIGURATION & CONSTANTS ======================
NUM_PLAYERS = 6
PLAYER_FEATURES = 13
GLOBAL_FEATURES = 9
TOTAL_FLAT_FEATURES = (NUM_PLAYERS * PLAYER_FEATURES) + GLOBAL_FEATURES

# Normalization bounds
POS_MIN_X, POS_MAX_X = -4096, 4096
POS_MIN_Y, POS_MAX_Y = -6000, 6000
POS_MIN_Z, POS_MAX_Z = 0, 2044
VEL_MIN, VEL_MAX = -2300, 2300
BOOST_MIN, BOOST_MAX = 0, 100
BALL_VEL_MIN, BALL_VEL_MAX = -6000, 6000
BOOST_PAD_MIN, BOOST_PAD_MAX = 0, 10
DIST_MIN, DIST_MAX = 0, (8192**2 + 10240**2 + 2044**2)**(1/2)

def normalize(val, min_val, max_val):
    """Normalizes a value to the [0, 1] range."""
    return (val - min_val) / (max_val - min_val + 1e-8)

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Rocket League Baseline MLP Training")
    parser.add_argument('--data-dir', type=str,
                        default="E:\\Raw RL Esports Replays\\Big Replay Dataset\\split_dataset",
                        help='Path to the parent directory containing train/val/test splits.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for training.')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--wandb-project', type=str, default="rl-goal-prediction-baseline", help="W&B project name.")
    parser.add_argument('--run-name', type=str, default=None, help="A custom name for the W&B run.")
    parser.add_argument('--resume', action='store_true', help='Resume training from a checkpoint.')
    parser.add_argument('--checkpoint-path', type=str, default='./baseline_checkpoint.pth', help='Path to save/load the latest checkpoint.')
    parser.add_argument('--checkpoint-every', type=int, default=5, help='Save a checkpoint every N epochs.')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of workers for DataLoader. Set to 0 to debug.')
    return parser.parse_args()


# ====================== ROBUST MEMORY-EFFICIENT DATASET CLASS V4 ======================
class RobustLazyDataset(Dataset):
    """
    V4 of the Dataset. Uses linecache for fast, random-access line reading,
    which is vastly more performant than pandas for this use case.
    """
    def __init__(self, list_of_csv_paths):
        self.csv_paths = list_of_csv_paths
        self.file_info = []
        self.cumulative_rows = [0]
        self.header = None
        
        print("--- Scanning dataset files to build index... ---")
        total_rows = 0
        for path in tqdm(self.csv_paths, desc="Indexing files"):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    if self.header is None:
                        # Store the header from the first file
                        self.header = f.readline().strip().split(',')
                    # Count remaining lines
                    num_lines = sum(1 for _ in f)
                if num_lines > 0:
                    self.file_info.append({'path': path, 'rows': num_lines})
                    total_rows += num_lines
                    self.cumulative_rows.append(total_rows)
            except Exception as e:
                print(f"\nWarning: Could not process file {path}. Skipping. Error: {e}")

        self.length = total_rows

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError("Index out of range")
        
        file_index = np.searchsorted(self.cumulative_rows, idx, side='right') - 1
        file_path = self.file_info[file_index]['path']
        local_idx = idx - self.cumulative_rows[file_index]

        try:
            # linecache is 1-based. Line 1 is the header. Data starts at line 2.
            # So we need to read line `local_idx + 2`.
            line = linecache.getline(file_path, local_idx + 2)
            if not line:
                return None

            # Create a dictionary from the header and the line values
            values = line.strip().split(',')
            row = dict(zip(self.header, values))

            player_features_flat = [item for i in range(NUM_PLAYERS) for item in [normalize(float(row[f'p{i}_pos_x']), POS_MIN_X, POS_MAX_X), normalize(float(row[f'p{i}_pos_y']), POS_MIN_Y, POS_MAX_Y), normalize(float(row[f'p{i}_pos_z']), POS_MIN_Z, POS_MAX_Z), normalize(float(row[f'p{i}_vel_x']), VEL_MIN, VEL_MAX), normalize(float(row[f'p{i}_vel_y']), VEL_MIN, VEL_MAX), normalize(float(row[f'p{i}_vel_z']), VEL_MIN, VEL_MAX), float(row[f'p{i}_forward_x']), float(row[f'p{i}_forward_y']), float(row[f'p{i}_forward_z']), normalize(float(row[f'p{i}_boost_amount']), BOOST_MIN, BOOST_MAX), float(row[f'p{i}_team']), float(row[f'p{i}_alive']), normalize(float(row[f'p{i}_dist_to_ball']), DIST_MIN, DIST_MAX)]]
            global_features = [normalize(float(row['ball_pos_x']), POS_MIN_X, POS_MAX_X), normalize(float(row['ball_pos_y']), POS_MIN_Y, POS_MAX_Y), normalize(float(row['ball_pos_z']), POS_MIN_Z, POS_MAX_Z), normalize(float(row['ball_vel_x']), BALL_VEL_MIN, BALL_VEL_MAX), normalize(float(row['ball_vel_y']), BALL_VEL_MIN, BALL_VEL_MAX), normalize(float(row['ball_vel_z']), BALL_VEL_MIN, BALL_VEL_MAX), normalize(float(row['boost_pad_0_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX), float(row['ball_hit_team_num']), normalize(min(float(row['seconds_remaining']), 300.0), 0, 300)]
            features = torch.tensor(player_features_flat + global_features, dtype=torch.float32)
            orange_y = torch.tensor([float(row['team_0_goal_in_event_window'])], dtype=torch.float32)
            blue_y = torch.tensor([float(row['team_1_goal_in_event_window'])], dtype=torch.float32)

            return features, orange_y, blue_y
        except (ValueError, KeyError, IndexError):
            # If line is malformed or there's a parsing error, skip it.
            return None

def collate_fn_skip_none(batch):
    """A custom collate_fn that filters out None values from a batch."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


# ====================== BASELINE MLP MODEL ======================
class BaselineMLP(nn.Module):
    def __init__(self, input_dim=TOTAL_FLAT_FEATURES):
        super(BaselineMLP, self).__init__(); self.body = nn.Sequential(nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5)); self.orange_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid()); self.blue_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
    def forward(self, x): x_p = self.body(x); return self.orange_head(x_p), self.blue_head(x_p)


# ====================== MAIN EXECUTION ======================
def main():
    args = parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"--- Using device: {device} ---")

    try:
        wandb.init(project=args.wandb_project, name=args.run_name, config=args, resume="allow")
        print("--- Weights & Biases successfully initialized ---")
    except Exception as e:
        print(f"--- Could not initialize W&B: {e}. Running without logging. ---")
        wandb.run = None

    print("\n--- Initializing Data Loaders (Robust, Fast Lazy Loading) ---")
    train_dir = os.path.join(args.data_dir, 'train'); val_dir = os.path.join(args.data_dir, 'val')
    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.csv')]
    val_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.csv')]
    
    train_dataset = RobustLazyDataset(train_files)
    val_dataset = RobustLazyDataset(val_files)

    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total validation samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_skip_none)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_skip_none)

    model = BaselineMLP().to(device); optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate); criterion = nn.BCELoss()

    start_epoch = 0; best_val_f1 = 0.0
    if args.resume:
        if os.path.exists(args.checkpoint_path):
            print(f"--- Resuming from checkpoint: {args.checkpoint_path} ---")
            checkpoint = torch.load(args.checkpoint_path, map_location=torch.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state']); optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint['epoch'] + 1; best_val_f1 = checkpoint.get('best_val_f1', 0.0)
            print(f"--- Resumed from epoch {start_epoch}. Best avg F1 so far: {best_val_f1:.4f} ---")
        else:
            print(f"--- WARNING: --resume flag was set, but no checkpoint found. Starting from scratch. ---")

    print(f"\n--- Starting Training from epoch {start_epoch + 1} to {args.epochs} ---")
    for epoch in range(start_epoch, args.epochs):
        model.train(); total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [TRAIN]")
        for batch in pbar:
            if batch is None: continue
            features, orange_labels, blue_labels = batch
            features, orange_labels, blue_labels = features.to(device), orange_labels.to(device), blue_labels.to(device)
            optimizer.zero_grad(); orange_pred, blue_pred = model(features); loss = criterion(orange_pred, orange_labels) + criterion(blue_pred, blue_labels); loss.backward(); optimizer.step(); total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0
        # Initialize all the lists needed for validation metrics
        val_probs_o, val_labels_o = [], []
        val_probs_b, val_labels_b = [], []
        val_preds_o_binary, val_preds_b_binary = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [VAL]"):
                if batch is None: continue
                features, orange_labels, blue_labels = batch
                features, orange_labels, blue_labels = features.to(device), orange_labels.to(device), blue_labels.to(device)
                orange_pred_prob, blue_pred_prob = model(features)
                total_val_loss += (criterion(orange_pred_prob, orange_labels) + criterion(blue_pred_prob, blue_labels)).item()
                val_probs_o.extend(orange_pred_prob.cpu().numpy().flatten()); val_labels_o.extend(orange_labels.cpu().numpy().flatten()); val_probs_b.extend(blue_pred_prob.cpu().numpy().flatten()); val_labels_b.extend(blue_labels.cpu().numpy().flatten()); val_preds_o_binary.extend((orange_pred_prob.cpu().numpy() > 0.5).astype(int).flatten()); val_preds_b_binary.extend((blue_pred_prob.cpu().numpy() > 0.5).astype(int).flatten())

        avg_train_loss = total_train_loss / len(train_loader); avg_val_loss = total_val_loss / len(val_loader)
        val_f1_o, val_f1_b = f1_score(val_labels_o, val_preds_o_binary, zero_division=0), f1_score(val_labels_b, val_preds_b_binary, zero_division=0)
        val_prec_o, val_recall_o = precision_score(val_labels_o, val_preds_o_binary, zero_division=0), recall_score(val_labels_o, val_preds_o_binary, zero_division=0)
        val_prec_b, val_recall_b = precision_score(val_labels_b, val_preds_b_binary, zero_division=0), recall_score(val_labels_b, val_preds_b_binary, zero_division=0)
        avg_val_f1 = (val_f1_o + val_f1_b) / 2
        print(f"\nEpoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Avg Val F1: {avg_val_f1:.4f}")
        
        if wandb.run:
            np_val_labels_o, np_val_probs_o, np_val_labels_b, np_val_probs_b, np_val_preds_o_binary, np_val_preds_b_binary, class_names = np.array(val_labels_o), np.array(val_probs_o), np.array(val_labels_b), np.array(val_probs_b), np.array(val_preds_o_binary), np.array(val_preds_b_binary), ["No Goal", "Goal"]
            wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss, "val_f1_orange": val_f1_o, "val_f1_blue": val_f1_b, "val_precision_orange": val_prec_o, "val_recall_orange": val_recall_o, "val_precision_blue": val_prec_b, "val_recall_blue": val_recall_b, "val_cm_orange": wandb.plot.confusion_matrix(y_true=np_val_labels_o, preds=np_val_preds_o_binary, class_names=class_names), "val_cm_blue": wandb.plot.confusion_matrix(y_true=np_val_labels_b, preds=np_val_preds_b_binary, class_names=class_names), "pr_curve_orange": wandb.plot.pr_curve(y_true=np_val_labels_o, y_probas=np.stack([1-np_val_probs_o, np_val_probs_o], axis=1), labels=class_names), "pr_curve_blue": wandb.plot.pr_curve(y_true=np_val_labels_b, y_probas=np.stack([1-np_val_probs_b, np_val_probs_b], axis=1), labels=class_names), "roc_curve_orange": wandb.plot.roc_curve(y_true=np_val_labels_o, y_probas=np.stack([1-np_val_probs_o, np_val_probs_o], axis=1), labels=class_names), "roc_curve_blue": wandb.plot.roc_curve(y_true=np_val_labels_b, y_probas=np.stack([1-np_val_probs_b, np_val_probs_b], axis=1), labels=class_names)})

        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1; best_model_path = os.path.join(os.path.dirname(args.checkpoint_path), 'best_model.pth')
            print(f"  *** New best model found (Avg F1: {best_val_f1:.4f}). Saving 'best' checkpoint to '{best_model_path}' ***")
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'best_val_f1': best_val_f1}, best_model_path)
        
        if (epoch + 1) % args.checkpoint_every == 0 or (epoch + 1) == args.epochs:
            print(f"--- Saving periodic checkpoint at epoch {epoch + 1} to '{args.checkpoint_path}' ---")
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'best_val_f1': best_val_f1, 'args': args}, args.checkpoint_path)

    print("\n--- Training Complete ---")
    
    # ====================== FINAL TEST EVALUATION ======================
    print("\n--- Running Final Evaluation on Best Model ---")
    best_model_path = os.path.join(os.path.dirname(args.checkpoint_path), 'best_model.pth')
    if not os.path.exists(best_model_path):
        print("--- No 'best_model.pth' found. Skipping final test evaluation. ---")
    else:
        print(f"--- Loading best model from: {best_model_path} ---")
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        test_dir = os.path.join(args.data_dir, 'test'); test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.csv')]
        if not test_files:
            print("--- No test files found. Skipping final evaluation. ---")
        else:
            test_dataset = RobustLazyDataset(test_files)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_skip_none)
            print(f"Total test samples: {len(test_dataset)}")
            model.eval(); test_preds_o_binary, test_labels_o, test_preds_b_binary, test_labels_b = [], [], [], []
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="[FINAL TEST]"):
                    if batch is None: continue
                    features, orange_labels, blue_labels = batch
                    features, orange_labels, blue_labels = features.to(device), orange_labels.to(device), blue_labels.to(device)
                    orange_pred, blue_pred = model(features)
                    test_preds_o_binary.extend((orange_pred.cpu().numpy() > 0.5).astype(int).flatten()); test_labels_o.extend(orange_labels.cpu().numpy().astype(int).flatten())
                    test_preds_b_binary.extend((blue_pred.cpu().numpy() > 0.5).astype(int).flatten()); test_labels_b.extend(blue_labels.cpu().numpy().astype(int).flatten())

            test_f1_o, test_f1_b = f1_score(test_labels_o, test_preds_o_binary, zero_division=0), f1_score(test_labels_b, test_preds_b_binary, zero_division=0)
            test_prec_o, test_recall_o = precision_score(test_labels_o, test_preds_o_binary, zero_division=0), recall_score(test_labels_o, test_preds_o_binary, zero_division=0)
            test_prec_b, test_recall_b = precision_score(test_labels_b, test_preds_b_binary, zero_division=0), recall_score(test_labels_b, test_preds_b_binary, zero_division=0)
            print("\n--- FINAL TEST RESULTS ---"); print(f"  Test F1 (Orange): {test_f1_o:.4f} | F1 (Blue): {test_f1_b:.4f}"); print(f"  Test Precision (Orange): {test_prec_o:.4f} | Recall (Orange): {test_recall_o:.4f}"); print(f"  Test Precision (Blue): {test_prec_b:.4f} | Recall (Blue): {test_recall_b:.4f}")
            if wandb.run:
                wandb.summary["best_epoch"] = checkpoint['epoch'] + 1; wandb.summary["test_f1_orange"] = test_f1_o; wandb.summary["test_f1_blue"] = test_f1_b
                wandb.summary["test_precision_orange"] = test_prec_o; wandb.summary["test_recall_orange"] = test_recall_o
                wandb.summary["test_precision_blue"] = test_prec_b; wandb.summary["test_recall_blue"] = test_recall_b
                wandb.log({"test_cm_orange": wandb.plot.confusion_matrix(y_true=np.array(test_labels_o), preds=np.array(test_preds_o_binary), class_names=["No Goal", "Goal"]), "test_cm_blue": wandb.plot.confusion_matrix(y_true=np.array(test_labels_b), preds=np.array(test_preds_b_binary), class_names=["No Goal", "Goal"])})

    if wandb.run: wandb.finish()
    print("\n--- Script Finished ---")

if __name__ == '__main__':
    main()