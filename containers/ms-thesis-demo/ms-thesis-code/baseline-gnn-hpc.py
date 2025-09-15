import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch.utils.data import Dataset 
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
import wandb
import linecache

# ====================== CONFIGURATION & CONSTANTS ======================
NUM_PLAYERS = 6
PLAYER_FEATURES = 13
HIDDEN_DIM = 32
NUM_TRACKED_BOOST_PADS = 6
GLOBAL_FEATURE_DIM = 3 + 3 + NUM_TRACKED_BOOST_PADS + 1 + 1 # ball_pos, ball_vel, pads, ball_hit, time

# Normalization bounds
POS_MIN_X, POS_MAX_X = -4096, 4096; POS_MIN_Y, POS_MAX_Y = -6000, 6000; POS_MIN_Z, POS_MAX_Z = 0, 2044
VEL_MIN, VEL_MAX = -2300, 2300; BOOST_MIN, BOOST_MAX = 0, 100; BALL_VEL_MIN, BALL_VEL_MAX = -6000, 6000
BOOST_PAD_MIN, BOOST_PAD_MAX = 0, 10; DIST_MIN, DIST_MAX = 0, (8192**2 + 10240**2 + 2044**2)**(1/2)

def normalize(val, min_val, max_val): return (val - min_val) / (max_val - min_val + 1e-8)

def parse_args():
    parser = argparse.ArgumentParser(description="Rocket League GCN Training")
    parser.add_argument('--data-dir', type=str, required=True, help='Parent directory of train/val/test splits.')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for GNNs.')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA.')
    parser.add_argument('--wandb-project', type=str, default="rl-goal-prediction-gcn", help="W&B project name.")
    parser.add_argument('--run-name', type=str, default=None, help="Custom name for the W&B run.")
    parser.add_argument('--resume', action='store_true', help='Resume training from a checkpoint.')
    parser.add_argument('--checkpoint-path', type=str, default='./gcn_checkpoint.pth', help='Path for periodic checkpoints.')
    parser.add_argument('--checkpoint-every', type=int, default=5, help='Save checkpoint every N epochs.')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for DataLoader. Start lower for GNNs.')
    return parser.parse_args()

# ====================== DATASET CLASS (GNN Version) ======================
class GNNLazyDataset(Dataset):
    def __init__(self, list_of_csv_paths):
        self.csv_paths = list_of_csv_paths; self.file_info = []; self.cumulative_rows = [0]; self.header = None; total_rows = 0
        print("--- Scanning dataset files to build index... ---")
        for path in tqdm(self.csv_paths, desc="Indexing files"):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    if self.header is None: self.header = f.readline().strip().split(',')
                    num_lines = sum(1 for _ in f)
                if num_lines > 0: self.file_info.append({'path': path, 'rows': num_lines}); total_rows += num_lines; self.cumulative_rows.append(total_rows)
            except Exception as e: print(f"\nWarning: Could not process file {path}. Skipping. Error: {e}")
        self.length = total_rows
    def __len__(self): return self.length
    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length: raise IndexError("Index out of range")
        file_index = np.searchsorted(self.cumulative_rows, idx, side='right') - 1; file_path = self.file_info[file_index]['path']; local_idx = idx - self.cumulative_rows[file_index]
        try:
            line = linecache.getline(file_path, local_idx + 2)
            if not line: return None
            row = dict(zip(self.header, line.strip().split(',')))
            
            # 1. Player (Node) Features
            x_features = []
            for i in range(NUM_PLAYERS):
                x_features.append([normalize(float(row[f'p{i}_pos_x']), POS_MIN_X, POS_MAX_X), normalize(float(row[f'p{i}_pos_y']), POS_MIN_Y, POS_MAX_Y), normalize(float(row[f'p{i}_pos_z']), POS_MIN_Z, POS_MAX_Z), normalize(float(row[f'p{i}_vel_x']), VEL_MIN, VEL_MAX), normalize(float(row[f'p{i}_vel_y']), VEL_MIN, VEL_MAX), normalize(float(row[f'p{i}_vel_z']), VEL_MIN, VEL_MAX), float(row[f'p{i}_forward_x']), float(row[f'p{i}_forward_y']), float(row[f'p{i}_forward_z']), normalize(float(row[f'p{i}_boost_amount']), BOOST_MIN, BOOST_MAX), float(row[f'p{i}_team']), float(row[f'p{i}_alive']), normalize(float(row[f'p{i}_dist_to_ball']), DIST_MIN, DIST_MAX)])
            x_tensor = torch.tensor(x_features, dtype=torch.float32)

            # 2. Global Features (with all boost pads)
            global_features_list = [normalize(float(row['ball_pos_x']), POS_MIN_X, POS_MAX_X), normalize(float(row['ball_pos_y']), POS_MIN_Y, POS_MAX_Y), normalize(float(row['ball_pos_z']), POS_MIN_Z, POS_MAX_Z), normalize(float(row['ball_vel_x']), BALL_VEL_MIN, BALL_VEL_MAX), normalize(float(row['ball_vel_y']), BALL_VEL_MIN, BALL_VEL_MAX), normalize(float(row['ball_vel_z']), BALL_VEL_MIN, BALL_VEL_MAX)]
            for pad_idx in range(NUM_TRACKED_BOOST_PADS):
                global_features_list.append(normalize(float(row.get(f'boost_pad_{pad_idx}_respawn', BOOST_PAD_MAX)), BOOST_PAD_MIN, BOOST_PAD_MAX))
            global_features_list.extend([float(row['ball_hit_team_num']), normalize(min(float(row['seconds_remaining']), 300.0), 0, 300)])
            global_features_tensor = torch.tensor(global_features_list, dtype=torch.float32)
            
            # 3. Graph Structure (Edges)
            edge_index_list, edge_weights_list = [], []
            for i in range(NUM_PLAYERS):
                for j in range(NUM_PLAYERS):
                    if i != j:
                        dist = torch.norm(x_tensor[i, :3] - x_tensor[j, :3])
                        weight = 1.0 / (1.0 + dist + 1e-8)
                        edge_weights_list.append(weight); edge_index_list.append([i, j])
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            edge_weights = torch.tensor(edge_weights_list, dtype=torch.float32)

            # 4. Labels
            orange_y = torch.tensor([float(row['team_1_goal_in_event_window'])], dtype=torch.float32)
            blue_y = torch.tensor([float(row['team_0_goal_in_event_window'])], dtype=torch.float32)
            
            return Data(x=x_tensor, edge_index=edge_index, edge_weight=edge_weights, global_features=global_features_tensor.unsqueeze(0), orange_y=orange_y, blue_y=blue_y)
        except (ValueError, KeyError, IndexError) as e: 
            # print(f"Skipping malformed row {idx}: {e}") # Optional: for debugging bad data
            return None

def pyg_collate_skip_none(batch_list):
    """
    A collate_fn for PyG DataLoader that filters out None items
    and then uses PyG's Batch.from_data_list to correctly form a batch.
    """
    # 1. Filter out None values from the list of Data objects
    batch_list = [item for item in batch_list if item is not None]

    # 2. If the batch is empty after filtering, return None.
    #    The training loop must handle this.
    if not batch_list:
        return None

    # 3. If there are valid items, use PyG's standard batching mechanism.
    return Batch.from_data_list(batch_list)

# ====================== GCN MODEL ======================
class SafeRocketLeagueGCN(nn.Module):
    def __init__(self):
        super(SafeRocketLeagueGCN, self).__init__()
        self.conv1 = GCNConv(PLAYER_FEATURES, HIDDEN_DIM); self.bn1 = nn.BatchNorm1d(HIDDEN_DIM)
        self.conv2 = GCNConv(HIDDEN_DIM, HIDDEN_DIM); self.bn2 = nn.BatchNorm1d(HIDDEN_DIM)
        # Dropout added for regularization, similar to the MLP baseline
        self.dropout = nn.Dropout(0.5)
        self.shared_body = nn.Sequential(nn.Linear(HIDDEN_DIM + GLOBAL_FEATURE_DIM, 128), nn.BatchNorm1d(128), nn.ReLU(), self.dropout, nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), self.dropout)
        self.orange_head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.blue_head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_weight))); x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_weight))); x = self.dropout(x)
        graph_embed = global_mean_pool(x, data.batch)
        combined_features = torch.cat([graph_embed, data.global_features], dim=1)
        shared_output = self.shared_body(combined_features)
        return self.orange_head(shared_output), self.blue_head(shared_output)

# ====================== MAIN EXECUTION ======================
def main():
    args = parse_args(); use_cuda = not args.no_cuda and torch.cuda.is_available(); device = torch.device("cuda" if use_cuda else "cpu"); print(f"--- Using device: {device} ---")

    start_epoch, best_val_f1, wandb_run_id = 0, 0.0, None
    if args.resume:
        if os.path.exists(args.checkpoint_path):
            print(f"--- Found checkpoint. Resuming: {args.checkpoint_path} ---")
            checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
            start_epoch = checkpoint['epoch'] + 1; best_val_f1 = checkpoint.get('best_val_f1', 0.0); wandb_run_id = checkpoint.get('wandb_run_id')
            print(f"--- Resuming from epoch {start_epoch}. Best F1: {best_val_f1:.4f} ---")
            if wandb_run_id: print(f"--- Resuming W&B run: {wandb_run_id} ---")
        else: print(f"--- WARNING: --resume set, but no checkpoint found. Starting fresh. ---")

    try: wandb.init(project=args.wandb_project, name=args.run_name, config=args, id=wandb_run_id, resume="allow"); print("--- W&B initialized ---")
    except Exception as e: print(f"--- W&B init failed: {e}. Running without logging. ---"); wandb.run = None

    print("\n--- Initializing Data Loaders (Lazy Loading for GNN) ---")
    train_dir = os.path.join(args.data_dir, 'train'); val_dir = os.path.join(args.data_dir, 'val')
    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.csv')]
    val_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.csv')]
    train_dataset = GNNLazyDataset(train_files); val_dataset = GNNLazyDataset(val_files)
    print(f"Total training samples: {len(train_dataset)}"); print(f"Total validation samples: {len(val_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=pyg_collate_skip_none)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=pyg_collate_skip_none)

    model = SafeRocketLeagueGCN().to(device); optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate); criterion = nn.BCELoss()
    if args.resume and 'checkpoint' in locals(): model.load_state_dict(checkpoint['model_state']); optimizer.load_state_dict(checkpoint['optimizer_state'])

    print(f"\n--- Starting Training from epoch {start_epoch + 1} to {args.epochs} ---")
    for epoch in range(start_epoch, args.epochs):
        model.train(); total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [TRAIN]")
        for batch in pbar:
            if batch is None: continue
            batch = batch.to(device) # PyG DataLoader batch is already a single Batch object
            orange_labels = batch.orange_y.view(-1, 1)
            blue_labels = batch.blue_y.view(-1, 1)
            optimizer.zero_grad(); orange_pred, blue_pred = model(batch); loss = criterion(orange_pred, orange_labels) + criterion(blue_pred, blue_labels); loss.backward(); optimizer.step(); total_train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        model.eval(); total_val_loss = 0; val_probs_o, val_labels_o, val_probs_b, val_labels_b, val_preds_o_binary, val_preds_b_binary = [], [], [], [], [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [VAL]"):
                if batch is None: continue
                batch = batch.to(device)
                orange_labels = batch.orange_y.view(-1, 1)
                blue_labels = batch.blue_y.view(-1, 1)
                orange_pred_prob, blue_pred_prob = model(batch); total_val_loss += (criterion(orange_pred_prob, orange_labels) + criterion(blue_pred_prob, blue_labels)).item()
                val_probs_o.extend(orange_pred_prob.cpu().numpy().flatten()); val_labels_o.extend(orange_labels.cpu().numpy().flatten()); val_probs_b.extend(blue_pred_prob.cpu().numpy().flatten()); val_labels_b.extend(blue_labels.cpu().numpy().flatten()); val_preds_o_binary.extend((orange_pred_prob.cpu().numpy() > 0.5).astype(int).flatten()); val_preds_b_binary.extend((blue_pred_prob.cpu().numpy() > 0.5).astype(int).flatten())

        avg_train_loss = total_train_loss / len(train_loader); avg_val_loss = total_val_loss / len(val_loader)
        val_f1_o, val_f1_b = f1_score(val_labels_o, val_preds_o_binary, zero_division=0), f1_score(val_labels_b, val_preds_b_binary, zero_division=0); avg_val_f1 = (val_f1_o + val_f1_b) / 2
        print(f"\nEpoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Avg Val F1: {avg_val_f1:.4f}")
        
        if wandb.run:
            np_val_labels_o, np_val_probs_o, np_val_labels_b, np_val_probs_b, np_val_preds_o_binary, np_val_preds_b_binary, class_names = np.array(val_labels_o), np.array(val_probs_o), np.array(val_labels_b), np.array(val_probs_b), np.array(val_preds_o_binary), np.array(val_preds_b_binary), ["No Goal", "Goal"]
            wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss, "val_f1_orange": val_f1_o, "val_f1_blue": val_f1_b, "avg_val_f1": avg_val_f1, "val_precision_orange": precision_score(val_labels_o, val_preds_o_binary, zero_division=0), "val_recall_orange": recall_score(val_labels_o, val_preds_o_binary, zero_division=0), "val_precision_blue": precision_score(val_labels_b, val_preds_b_binary, zero_division=0), "val_recall_blue": recall_score(val_labels_b, val_preds_b_binary, zero_division=0), "val_cm_orange": wandb.plot.confusion_matrix(y_true=np_val_labels_o, preds=np_val_preds_o_binary, class_names=class_names), "val_cm_blue": wandb.plot.confusion_matrix(y_true=np_val_labels_b, preds=np_val_preds_b_binary, class_names=class_names), "pr_curve_orange": wandb.plot.pr_curve(y_true=np_val_labels_o, y_probas=np.stack([1-np_val_probs_o, np_val_probs_o], axis=1), labels=class_names), "pr_curve_blue": wandb.plot.pr_curve(y_true=np_val_labels_b, y_probas=np.stack([1-np_val_probs_b, np_val_probs_b], axis=1), labels=class_names), "roc_curve_orange": wandb.plot.roc_curve(y_true=np_val_labels_o, y_probas=np.stack([1-np_val_probs_o, np_val_probs_o], axis=1), labels=class_names), "roc_curve_blue": wandb.plot.roc_curve(y_true=np_val_labels_b, y_probas=np.stack([1-np_val_probs_b, np_val_probs_b], axis=1), labels=class_names)})

        current_wandb_id = wandb.run.id if wandb.run else None
        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1; best_model_path = os.path.join(os.path.dirname(args.checkpoint_path), 'best_gcn_model.pth')
            print(f"  *** New best model (Avg F1: {best_val_f1:.4f}). Saving 'best' checkpoint. ***")
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'best_val_f1': best_val_f1, 'wandb_run_id': current_wandb_id}, best_model_path)
        
        if (epoch + 1) % args.checkpoint_every == 0 or (epoch + 1) == args.epochs:
            print(f"--- Saving periodic checkpoint at epoch {epoch + 1}. ---")
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'best_val_f1': best_val_f1, 'args': args, 'wandb_run_id': current_wandb_id}, args.checkpoint_path)

    print("\n--- Training Complete ---")
    
    # FINAL TEST EVALUATION
    print("\n--- Running Final Evaluation on Best GCN Model ---")
    best_model_path = os.path.join(os.path.dirname(args.checkpoint_path), 'best_gcn_model.pth')
    if not os.path.exists(best_model_path): print("--- No 'best_gcn_model.pth' found. Skipping final test evaluation. ---")
    else:
        print(f"--- Loading best model from: {best_model_path} ---")
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False); model.load_state_dict(checkpoint['model_state'])
        test_dir = os.path.join(args.data_dir, 'test'); test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.csv')]
        if not test_files: print("--- No test files found. ---")
        else:
            test_dataset = GNNLazyDataset(test_files)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=pyg_collate_skip_none)
            print(f"Total test samples: {len(test_dataset)}")
            model.eval(); test_preds_o_binary, test_labels_o, test_preds_b_binary, test_labels_b = [], [], [], []
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="[FINAL TEST]"):
                    if batch is None: continue
                    batch = batch.to(device)
                    orange_labels = batch.orange_y.view(-1, 1)
                    blue_labels = batch.blue_y.view(-1, 1)
                    orange_pred, blue_pred = model(batch)
                    test_preds_o_binary.extend((orange_pred.cpu().numpy() > 0.5).astype(int).flatten()); test_labels_o.extend(batch.orange_y.cpu().numpy().astype(int).flatten())
                    test_preds_b_binary.extend((blue_pred.cpu().numpy() > 0.5).astype(int).flatten()); test_labels_b.extend(batch.blue_y.cpu().numpy().astype(int).flatten())

            test_f1_o, test_f1_b = f1_score(test_labels_o, test_preds_o_binary, zero_division=0), f1_score(test_labels_b, test_preds_b_binary, zero_division=0)
            print("\n--- FINAL TEST RESULTS ---"); print(f"  Test F1 (Orange): {test_f1_o:.4f} | F1 (Blue): {test_f1_b:.4f}")
            if wandb.run:
                wandb.summary["best_epoch"] = checkpoint['epoch'] + 1; wandb.summary["test_f1_orange"] = test_f1_o; wandb.summary["test_f1_blue"] = test_f1_b
                wandb.log({"test_cm_orange": wandb.plot.confusion_matrix(y_true=np.array(test_labels_o), preds=np.array(test_preds_o_binary), class_names=["No Goal", "Goal"]), "test_cm_blue": wandb.plot.confusion_matrix(y_true=np.array(test_labels_b), preds=np.array(test_preds_b_binary), class_names=["No Goal", "Goal"])})

    if wandb.run: wandb.finish()
    print("\n--- Script Finished ---")

if __name__ == '__main__':
    main()