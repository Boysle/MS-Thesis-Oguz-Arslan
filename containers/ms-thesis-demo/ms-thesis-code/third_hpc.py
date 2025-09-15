import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import wandb
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ====================== CONFIGURATION ======================
NUM_PLAYERS = 6
PLAYER_FEATURES = 13  # xyz_pos, xyz_vel, xyz_forward, boost, team, alive, distance_to_ball
HIDDEN_DIM = 32
GLOBAL_FEATURE_DIM = 9  # xyz_ball_pos, xyz_ball_vel, boost_pad_respawn_times, ball_hit_team_num, seconds_remaining

# Normalization bounds
POS_MIN_X, POS_MAX_X = -4096, 4096
POS_MIN_Y, POS_MAX_Y = -6000, 6000
POS_MIN_Z, POS_MAX_Z = 0, 2044
VEL_MIN, VEL_MAX = -2300, 2300
BOOST_MIN, BOOST_MAX = 0, 100
BALL_VEL_MIN, BALL_VEL_MAX = -6000, 6000
BOOST_PAD_MIN, BOOST_PAD_MAX = 0, 10
DIST_MIN, DIST_MAX = 0, (8192**2 + 10240**2 + 2044**2)**(1/2)

# Feature Names for Visualization (ensure order matches PLAYER_FEATURES and GLOBAL_FEATURE_DIM)
PLAYER_FEATURE_NAMES = [
    'p_pos_x', 'p_pos_y', 'p_pos_z',
    'p_vel_x', 'p_vel_y', 'p_vel_z',
    'p_forward_x', 'p_forward_y', 'p_forward_z',
    'p_boost_amount', 'p_team', 'p_alive', 'p_dist_to_ball'
]
GLOBAL_FEATURE_NAMES = [
    'ball_pos_x', 'ball_pos_y', 'ball_pos_z',
    'ball_vel_x', 'ball_vel_y', 'ball_vel_z',
    'boost_pad_0_respawn', 'ball_hit_team_num', 'seconds_remaining'
]


def parse_args():
    parser = argparse.ArgumentParser(description="Rocket League GCN")
    parser.add_argument('--csv-path', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--checkpoint-path',
                      type=str,
                      default="/users/oguz.arslan1/MS-Thesis-Oguz-Arslan/checkpoints/model_checkpoint.pth",
                      help='Path to save/load checkpoints')
    parser.add_argument('--resume',
                      action='store_true',
                      help='Resume from checkpoint')
    return parser.parse_args()


# ====================== DATA LOADING ======================
def normalize(val, min_val, max_val):
    return (val - min_val) / (max_val - min_val + 1e-8)

def load_and_process_data(csv_path):
    print(f"[DATA] Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[DATA] Found {len(df)} raw samples")

    # Input validation
    required_columns = []
    for i in range(NUM_PLAYERS):
        required_columns.extend([
            f'p{i}_pos_x', f'p{i}_pos_y', f'p{i}_pos_z',
            f'p{i}_vel_x', f'p{i}_vel_y', f'p{i}_vel_z',
            f'p{i}_forward_x', f'p{i}_forward_y', f'p{i}_forward_z',
            f'p{i}_boost_amount', f'p{i}_team', f'p{i}_alive',
            f'p{i}_dist_to_ball'
        ])

    required_columns.extend([
        'ball_pos_x', 'ball_pos_y', 'ball_pos_z',
        'ball_vel_x', 'ball_vel_y', 'ball_vel_z',
        'boost_pad_0_respawn', 'ball_hit_team_num', 'seconds_remaining',
        'team_0_goal_prev_5s', 'team_1_goal_prev_5s' # Corrected understanding: these are next 5s
    ])

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")

    dataset = []

    for idx, row in df.iterrows():
        if row.isnull().any():
            continue
        if idx % 1000 == 0 and idx > 0:
            print(f"[PROGRESS] Processed {idx}/{len(df)} samples")

        x_features = []
        for i in range(NUM_PLAYERS):
            pos_x = normalize(float(row[f'p{i}_pos_x']), POS_MIN_X, POS_MAX_X)
            pos_y = normalize(float(row[f'p{i}_pos_y']), POS_MIN_Y, POS_MAX_Y)
            pos_z = normalize(float(row[f'p{i}_pos_z']), POS_MIN_Z, POS_MAX_Z)
            vel_x = normalize(float(row[f'p{i}_vel_x']), VEL_MIN, VEL_MAX)
            vel_y = normalize(float(row[f'p{i}_vel_y']), VEL_MIN, VEL_MAX)
            vel_z = normalize(float(row[f'p{i}_vel_z']), VEL_MIN, VEL_MAX)
            forward_x = float(row[f'p{i}_forward_x']) # Assuming forward vectors are already normalized or their magnitude is not critical
            forward_y = float(row[f'p{i}_forward_y'])
            forward_z = float(row[f'p{i}_forward_z'])
            boost = normalize(float(row[f'p{i}_boost_amount']), BOOST_MIN, BOOST_MAX)
            team = float(row[f'p{i}_team']) # Team is 0 or 1, not normalized
            alive = float(row[f'p{i}_alive']) # Alive is 0 or 1
            dist_to_ball = normalize(float(row[f'p{i}_dist_to_ball']), DIST_MIN, DIST_MAX)

            x_features.append([
                pos_x, pos_y, pos_z,
                vel_x, vel_y, vel_z,
                forward_x, forward_y, forward_z,
                boost, team, alive, dist_to_ball
            ])

        x_tensor = torch.tensor(x_features, dtype=torch.float32)
        assert x_tensor.shape == (NUM_PLAYERS, PLAYER_FEATURES), \
            f"Player features shape mismatch. Expected {(NUM_PLAYERS, PLAYER_FEATURES)}, got {x_tensor.shape}"

        edge_index_list = []
        edge_weights_list = []
        for i in range(NUM_PLAYERS):
            for j in range(NUM_PLAYERS):
                if i != j:
                    # Use normalized positions for distance calculation if x_tensor contains normalized pos
                    dist = torch.norm(x_tensor[i, :3] - x_tensor[j, :3])
                    weight = 1.0 / (1.0 + dist)
                    edge_weights_list.append(weight if not torch.isnan(weight) else 0.0)
                    edge_index_list.append([i, j])

        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_weights = torch.tensor(edge_weights_list, dtype=torch.float32)

        seconds_remaining_val = float(row['seconds_remaining'])
        normalized_seconds = normalize(min(seconds_remaining_val, 300.0), 0, 300) # Cap at 300s for normalization

        global_features_tensor = torch.tensor([
            normalize(float(row['ball_pos_x']), POS_MIN_X, POS_MAX_X),
            normalize(float(row['ball_pos_y']), POS_MIN_Y, POS_MAX_Y),
            normalize(float(row['ball_pos_z']), POS_MIN_Z, POS_MAX_Z),
            normalize(float(row['ball_vel_x']), BALL_VEL_MIN, BALL_VEL_MAX),
            normalize(float(row['ball_vel_y']), BALL_VEL_MIN, BALL_VEL_MAX),
            normalize(float(row['ball_vel_z']), BALL_VEL_MIN, BALL_VEL_MAX),
            normalize(float(row['boost_pad_0_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX), # Only one boost pad here?
            float(row['ball_hit_team_num']), # 0 or 1
            normalized_seconds
        ], dtype=torch.float32)

        assert global_features_tensor.shape == (GLOBAL_FEATURE_DIM,), \
            f"Global features shape mismatch. Expected {(GLOBAL_FEATURE_DIM,)}, got {global_features_tensor.shape}"

        # Target labels: "goal scored by team X in the NEXT 5 seconds"
        orange_y_tensor = torch.tensor([float(row['team_0_goal_prev_5s'])], dtype=torch.float32)
        blue_y_tensor = torch.tensor([float(row['team_1_goal_prev_5s'])], dtype=torch.float32)

        data = Data(
            x=x_tensor,
            edge_index=edge_index,
            edge_weight=edge_weights,
            global_features=global_features_tensor.unsqueeze(0), # Add batch dimension for consistency
            orange_y=orange_y_tensor,
            blue_y=blue_y_tensor
        )
        dataset.append(data)

    print(f"[DATA] Created {len(dataset)} valid graph samples")
    print(f"[DATA] Sample breakdown - Orange goals: {sum(d.orange_y.item() for d in dataset)}, "
          f"Blue goals: {sum(d.blue_y.item() for d in dataset)}")
    return dataset


# ====================== MODEL ======================
class SafeRocketLeagueGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(PLAYER_FEATURES, HIDDEN_DIM)
        self.bn1 = nn.BatchNorm1d(HIDDEN_DIM)
        self.conv2 = GCNConv(HIDDEN_DIM, HIDDEN_DIM)
        self.bn2 = nn.BatchNorm1d(HIDDEN_DIM)
        self.orange_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM + GLOBAL_FEATURE_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.blue_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM + GLOBAL_FEATURE_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        # GCNConv expects node features of shape [num_nodes, num_node_features]
        # If x is batched, global_mean_pool handles it with data.batch
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Pool node features to get graph-level embeddings
        graph_embed = global_mean_pool(x, data.batch) # data.batch is crucial for batched graphs

        # Concatenate graph embedding with global features
        # data.global_features is expected to be [batch_size, GLOBAL_FEATURE_DIM]
        # graph_embed is [batch_size, HIDDEN_DIM]
        combined_features = torch.cat([graph_embed, data.global_features], dim=1)
        
        return self.orange_head(combined_features), self.blue_head(combined_features)


# ====================== LOGGING & VISUALIZATION ======================
def log_feature_importance_to_wandb(node_feature_grads, global_feature_grads, epoch):
    if wandb.run: # Check if wandb is active
        wandb.log({
            'epoch': epoch,
            'node_feature_importance': wandb.Histogram(node_feature_grads.cpu().numpy()),
            'global_feature_importance': wandb.Histogram(global_feature_grads.cpu().numpy()),
        })

def plot_feature_distributions(samples_dict, feature_name_display,
                               feature_type='player', player_idx=None, feature_idx=None,
                               wandb_log_name=None):
    plt.figure(figsize=(12, 7))
    any_data_plotted = False
    for outcome, samples in samples_dict.items():
        if not samples:
            continue
        
        values = []
        if feature_type == 'player':
            if player_idx is not None: # Specific player
                values = [s['player_features'][player_idx, feature_idx] for s in samples if s['player_features'].shape[0] > player_idx and s['player_features'].shape[1] > feature_idx]
            else: # Average over players
                values = [np.mean(s['player_features'][:, feature_idx]) for s in samples if s['player_features'].shape[1] > feature_idx]
        elif feature_type == 'global':
            values = [s['global_features'][feature_idx] for s in samples if len(s['global_features']) > feature_idx]
        
        if values:
            sns.histplot(values, label=f'{outcome} ({len(values)})', kde=True, stat="density", common_norm=False, element="step")
            any_data_plotted = True

    if not any_data_plotted:
        plt.close() # Close if nothing was plotted
        return

    plt.title(f'Distribution of {feature_name_display}')
    plt.xlabel(feature_name_display + " (Normalized)")
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    if wandb.run and wandb_log_name:
        wandb.log({wandb_log_name: wandb.Image(plt)})
    plt.show()
    plt.close()

def plot_avg_positions(samples_list, plot_title_suffix, team_id_feature_idx, wandb_log_name=None):
    if not samples_list:
        print(f"No samples to plot for {plot_title_suffix}")
        return

    avg_player_pos = np.zeros((NUM_PLAYERS, 2)) # X, Y
    avg_ball_pos = np.zeros(2) # X, Y
    num_samples = len(samples_list)

    # Accumulate positions
    for sample_data in samples_list:
        # player_features are [NUM_PLAYERS, PLAYER_FEATURES]
        # global_features are [GLOBAL_FEATURE_DIM] (after flatten)
        avg_player_pos += sample_data['player_features'][:, :2] # Sum X, Y for all players (indices 0, 1)
        avg_ball_pos += sample_data['global_features'][:2]      # Sum ball X, Y (indices 0, 1)

    avg_player_pos /= num_samples
    avg_ball_pos /= num_samples

    plt.figure(figsize=(7, 9)) # Adjusted for typical Rocket League field proportions
    
    # Field boundaries (normalized 0-1 based on your POS_MIN/MAX)
    # These are just visual aids; actual field drawing would be more complex.
    plt.axhline(y=normalize(POS_MAX_Y, POS_MIN_Y, POS_MAX_Y) - 0.02, color='orange', linestyle='--', xmin=0.35, xmax=0.65, lw=2, label="Orange Goal Area") # Approx goal line
    plt.axhline(y=normalize(POS_MIN_Y, POS_MIN_Y, POS_MAX_Y) + 0.02, color='blue', linestyle='--', xmin=0.35, xmax=0.65, lw=2, label="Blue Goal Area")   # Approx goal line
    plt.xlim(normalize(POS_MIN_X,POS_MIN_X,POS_MAX_X) - 0.05, normalize(POS_MAX_X,POS_MIN_X,POS_MAX_X) + 0.05)
    plt.ylim(normalize(POS_MIN_Y,POS_MIN_Y,POS_MAX_Y) - 0.05, normalize(POS_MAX_Y,POS_MIN_Y,POS_MAX_Y) + 0.05)

    # Plot players
    # Team ID is used to color players. Assumes 'p_team' is 0 for blue, 1 for orange.
    player_colors = {0.0: 'blue', 1.0: 'orange'} # Mapping team ID to color

    for i in range(NUM_PLAYERS):
        # Get team for this player from the first sample (assuming it's consistent for player slot i)
        # This is a simplification. A robust way would be to average team ID if it could vary.
        # Player team is not normalized, it's 0 or 1.
        player_team_id = samples_list[0]['player_features'][i, team_id_feature_idx]
        color = player_colors.get(player_team_id, 'gray') # Default to gray if team ID unknown

        plt.scatter(avg_player_pos[i, 0], avg_player_pos[i, 1],
                    s=120, label=f'P{i}', color=color, edgecolors='black', alpha=0.8)
                    
    plt.scatter(avg_ball_pos[0], avg_ball_pos[1], s=180, color='darkgreen', marker='o', label='Ball', edgecolors='black', alpha=0.9)

    plt.title(f'Avg Positions: {plot_title_suffix} ({num_samples} samples)')
    plt.xlabel('X Position (Normalized)')
    plt.ylabel('Y Position (Normalized)')
    plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    
    if wandb.run and wandb_log_name:
        wandb.log({wandb_log_name: wandb.Image(plt)})
    plt.show()
    plt.close()

def log_misclassified_samples_table(samples_list, table_name_suffix, num_samples_to_log=5):
    if not samples_list or not wandb.run:
        return

    table_data = []
    # Sort by predicted probability to see "strongest" misclassifications
    # For FNs, sort by prob (closer to 0 is worse)
    # For FPs, sort by 1-prob (closer to 0, i.e. prob closer to 1, is worse)
    if "FN" in table_name_suffix: # False Negative: true=1, pred_prob low
        sorted_samples = sorted(samples_list, key=lambda s: s['predicted_prob'])
    elif "FP" in table_name_suffix: # False Positive: true=0, pred_prob high
        sorted_samples = sorted(samples_list, key=lambda s: s['predicted_prob'], reverse=True)
    else:
        sorted_samples = samples_list

    dist_to_ball_idx = PLAYER_FEATURE_NAMES.index('p_dist_to_ball')
    ball_pos_x_idx = GLOBAL_FEATURE_NAMES.index('ball_pos_x')
    ball_pos_y_idx = GLOBAL_FEATURE_NAMES.index('ball_pos_y')
    ball_vel_x_idx = GLOBAL_FEATURE_NAMES.index('ball_vel_x')
    seconds_remaining_idx = GLOBAL_FEATURE_NAMES.index('seconds_remaining')

    for i, sample in enumerate(sorted_samples[:num_samples_to_log]):
        row = [
            i,
            f"{sample['predicted_prob']:.3f}",
            sample['true_label'],
            f"{sample['global_features'][ball_pos_x_idx]:.2f}",
            f"{sample['global_features'][ball_pos_y_idx]:.2f}",
            f"{sample['global_features'][ball_vel_x_idx]:.2f}",
            f"{sample['global_features'][seconds_remaining_idx]:.2f}",
            f"{np.mean(sample['player_features'][:, dist_to_ball_idx]):.2f}"
        ]
        table_data.append(row)

    columns = ["#", "Pred Prob", "True Label", "Ball X", "Ball Y", "Ball Vel X", "Secs Rem", "Avg P_DistBall"]
    misclass_table = wandb.Table(data=table_data, columns=columns)
    wandb.log({f"{table_name_suffix}_examples": misclass_table})


# ====================== TRAINING ======================
def main():
    print("[STATUS] Starting script execution")
    args = parse_args()
    print(f"[CONFIG] Loaded configuration: {vars(args)}")

    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Using device: {device}")
    if str(device) == 'cuda':
        print("[OPTIM] Enabling CUDA optimizations")
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')

    try:
        if os.getenv("WANDB_API_KEY"):
             wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(project="rocket-league-gcn-safe", config=args, name=f"run_epochs{args.epochs}_bs{args.batch_size}_lr{args.learning_rate}")
        print("[LOGGING] Successfully initialized Weights & Biases")
    except Exception as e:
        print(f"[WARNING] Failed to initialize WandB: {str(e)}. Will proceed without WandB logging.")
        wandb.run = None # Ensure wandb.log calls don't crash

    dataset = load_and_process_data(args.csv_path)
    if not dataset:
        print("[ERROR] No data loaded. Exiting.")
        return
        
    train_data, test_data = train_test_split(dataset, test_size=args.test_size, random_state=args.random_seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True) # drop_last for stable batch norm
    test_loader = DataLoader(test_data, batch_size=args.batch_size, drop_last=False)

    model = SafeRocketLeagueGCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCELoss() # For use with Sigmoid outputs

    start_epoch = 0
    best_f1 = 0 # Example for saving best model, not fully implemented here

    if args.resume and os.path.exists(args.checkpoint_path):
        try:
            checkpoint = torch.load(args.checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint['epoch'] + 1
            best_f1 = checkpoint.get('best_f1', 0) # Use .get for backward compatibility
            print(f"[CHECKPOINT] Resuming from {args.checkpoint_path} at epoch {start_epoch}")
        except Exception as e:
            print(f"[ERROR] Could not load checkpoint: {e}. Starting from scratch.")
            start_epoch = 0
            best_f1 = 0


    for epoch in range(start_epoch, args.epochs):
        print(f"\n[EPOCH] Starting epoch {epoch+1}/{args.epochs}")
        model.train()
        total_loss = 0
        
        all_orange_preds_epoch, all_orange_labels_epoch = [], []
        all_blue_preds_epoch, all_blue_labels_epoch = [], []

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            if args.debug and epoch == start_epoch and batch_idx == 0: # For feature importance debugging
                batch.x.requires_grad_(True)
                if batch.global_features is not None:
                     batch.global_features.requires_grad_(True)

            optimizer.zero_grad()
            orange_pred_prob, blue_pred_prob = model(batch) # Probabilities from Sigmoid

            # Ensure targets are correctly shaped [batch_size, 1]
            loss_orange = criterion(orange_pred_prob, batch.orange_y.view_as(orange_pred_prob))
            loss_blue = criterion(blue_pred_prob, batch.blue_y.view_as(blue_pred_prob))
            loss = loss_orange + loss_blue
            
            loss.backward()

            if args.debug and epoch == start_epoch and batch_idx == 0: # Log grads only once for debugging
                if batch.x.grad is not None and batch.global_features is not None and batch.global_features.grad is not None:
                    node_feature_grads = batch.x.grad.abs().mean(dim=[0,1] if batch.x.grad.ndim == 3 else 0) # Handle batched node features correctly
                    global_feature_grads = batch.global_features.grad.abs().mean(dim=0)
                    log_feature_importance_to_wandb(node_feature_grads, global_feature_grads, epoch)
                else:
                    print(f"Warning: No gradients for features in epoch {epoch}, batch {batch_idx}")
                batch.x.requires_grad_(False) # Detach for next iterations
                if batch.global_features is not None:
                    batch.global_features.requires_grad_(False)

            optimizer.step()
            total_loss += loss.item()

            # Store predictions and labels for epoch metrics
            all_orange_preds_epoch.extend((orange_pred_prob.detach().cpu().numpy() > 0.5).astype(int).flatten())
            all_orange_labels_epoch.extend(batch.orange_y.cpu().numpy().flatten())
            all_blue_preds_epoch.extend((blue_pred_prob.detach().cpu().numpy() > 0.5).astype(int).flatten())
            all_blue_labels_epoch.extend(batch.blue_y.cpu().numpy().flatten())

        # Calculate and log training metrics for the epoch
        avg_train_loss = total_loss / len(train_loader)
        train_f1_orange = f1_score(all_orange_labels_epoch, all_orange_preds_epoch, zero_division=0)
        train_f1_blue = f1_score(all_blue_labels_epoch, all_blue_preds_epoch, zero_division=0)
        train_prec_orange = precision_score(all_orange_labels_epoch, all_orange_preds_epoch, zero_division=0)
        train_prec_blue = precision_score(all_blue_labels_epoch, all_blue_preds_epoch, zero_division=0)
        train_recall_orange = recall_score(all_orange_labels_epoch, all_orange_preds_epoch, zero_division=0)
        train_recall_blue = recall_score(all_blue_labels_epoch, all_blue_preds_epoch, zero_division=0)
        
        print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} | "
              f"F1: O {train_f1_orange:.3f}, B {train_f1_blue:.3f} | "
              f"Prec: O {train_prec_orange:.3f}, B {train_prec_blue:.3f} | "
              f"Recall: O {train_recall_orange:.3f}, B {train_recall_blue:.3f}")

        if wandb.run:
            wandb.log({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'train_f1_orange': train_f1_orange, 'train_f1_blue': train_f1_blue,
                'train_precision_orange': train_prec_orange, 'train_precision_blue': train_prec_blue,
                'train_recall_orange': train_recall_orange, 'train_recall_blue': train_recall_blue,
            })

            # Log confusion matrices (optional, can be verbose per epoch)
            if epoch % 5 == 0 or epoch == args.epochs -1 : # Log every 5 epochs
                cm_orange_train = confusion_matrix(all_orange_labels_epoch, all_orange_preds_epoch)
                cm_blue_train = confusion_matrix(all_blue_labels_epoch, all_blue_preds_epoch)
                wandb.log({
                    'train_cm_orange': wandb.plot.confusion_matrix(
                        y_true=np.array(all_orange_labels_epoch), preds=np.array(all_orange_preds_epoch), class_names=["No Goal", "Goal"]),
                    'train_cm_blue': wandb.plot.confusion_matrix(
                        y_true=np.array(all_blue_labels_epoch), preds=np.array(all_blue_preds_epoch), class_names=["No Goal", "Goal"])
                })


        if (epoch + 1) % 5 == 0:  # Save checkpoint periodically
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_f1': best_f1, # Update best_f1 based on validation if available
                'args': vars(args)
            }, args.checkpoint_path)
            print(f"[CHECKPOINT] Saved at epoch {epoch+1}")

    # ======== TEST EVALUATION & VISUALIZATION ========
    print("\n[TEST] Evaluating on test set and visualizing errors...")
    model.eval()
    
    all_test_orange_true_labels, all_test_orange_pred_probs, all_test_orange_pred_labels = [], [], []
    all_test_blue_true_labels, all_test_blue_pred_probs, all_test_blue_pred_labels = [], [], []
    
    # Store full input data for misclassification analysis
    # Each element: {'player_features': ndarray, 'global_features': ndarray, 'true_label': float, 'predicted_prob': float, 'predicted_label': float}
    all_input_data_orange_eval = []
    all_input_data_blue_eval = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = batch.to(device)
            orange_pred_prob, blue_pred_prob = model(batch) # Probabilities

            orange_preds_binary = (orange_pred_prob.cpu().numpy() > 0.5).astype(int)
            blue_preds_binary = (blue_pred_prob.cpu().numpy() > 0.5).astype(int)

            all_test_orange_true_labels.extend(batch.orange_y.cpu().numpy().flatten())
            all_test_orange_pred_probs.extend(orange_pred_prob.cpu().numpy().flatten())
            all_test_orange_pred_labels.extend(orange_preds_binary.flatten())

            all_test_blue_true_labels.extend(batch.blue_y.cpu().numpy().flatten())
            all_test_blue_pred_probs.extend(blue_pred_prob.cpu().numpy().flatten())
            all_test_blue_pred_labels.extend(blue_preds_binary.flatten())
            
            # Store input data for each sample in the batch
            for i in range(batch.num_graphs):
                data_sample = batch[i] # Individual Data object
                all_input_data_orange_eval.append({
                    'player_features': data_sample.x.cpu().numpy(),
                    'global_features': data_sample.global_features.cpu().numpy().flatten(), # Flatten if it was unsqueezed
                    'true_label': data_sample.orange_y.item(),
                    'predicted_prob': orange_pred_prob[i].item(),
                    'predicted_label': orange_preds_binary[i].item()
                })
                all_input_data_blue_eval.append({
                    'player_features': data_sample.x.cpu().numpy(),
                    'global_features': data_sample.global_features.cpu().numpy().flatten(),
                    'true_label': data_sample.blue_y.item(),
                    'predicted_prob': blue_pred_prob[i].item(),
                    'predicted_label': blue_preds_binary[i].item()
                })

    # Calculate final metrics
    test_f1_orange = f1_score(all_test_orange_true_labels, all_test_orange_pred_labels, zero_division=0)
    test_f1_blue = f1_score(all_test_blue_true_labels, all_test_blue_pred_labels, zero_division=0)
    cm_orange_test = confusion_matrix(all_test_orange_true_labels, all_test_orange_pred_labels)
    cm_blue_test = confusion_matrix(all_test_blue_true_labels, all_test_blue_pred_labels)

    print(f"\n[TEST] Final Metrics:")
    print(f"Orange F1: {test_f1_orange:.4f} | Blue F1: {test_f1_blue:.4f}")
    print(f"Orange CM:\n{cm_orange_test}")
    print(f"Blue CM:\n{cm_blue_test}")

    if wandb.run:
        wandb.log({
            'test_f1_orange': test_f1_orange, 'test_f1_blue': test_f1_blue,
            'test_cm_orange': wandb.plot.confusion_matrix(
                y_true=np.array(all_test_orange_true_labels), preds=np.array(all_test_orange_pred_labels), class_names=["No Goal", "Goal"]),
            'test_cm_blue': wandb.plot.confusion_matrix(
                y_true=np.array(all_test_blue_true_labels), preds=np.array(all_test_blue_pred_labels), class_names=["No Goal", "Goal"])
        })

    # --- Error Analysis & Visualization ---
    # Identify TP, FP, TN, FN samples
    true_o = np.array(all_test_orange_true_labels)
    pred_o = np.array(all_test_orange_pred_labels)
    tp_orange_samples = [s for i, s in enumerate(all_input_data_orange_eval) if true_o[i] == 1 and pred_o[i] == 1]
    fp_orange_samples = [s for i, s in enumerate(all_input_data_orange_eval) if true_o[i] == 0 and pred_o[i] == 1]
    fn_orange_samples = [s for i, s in enumerate(all_input_data_orange_eval) if true_o[i] == 1 and pred_o[i] == 0]
    tn_orange_samples = [s for i, s in enumerate(all_input_data_orange_eval) if true_o[i] == 0 and pred_o[i] == 0]
    
    # (Repeat for Blue team if desired)

    # Plot Feature Distributions for Orange Team Errors
    orange_samples_for_plotting = {"TP": tp_orange_samples, "FN": fn_orange_samples, "FP": fp_orange_samples, "TN": tn_orange_samples}
    
    dist_to_ball_idx = PLAYER_FEATURE_NAMES.index('p_dist_to_ball')
    plot_feature_distributions(orange_samples_for_plotting, 'Avg Player Dist to Ball (Orange)',
                               feature_type='player', feature_idx=dist_to_ball_idx,
                               wandb_log_name='orange_err_dist_player_dist_ball')
    
    boost_idx = PLAYER_FEATURE_NAMES.index('p_boost_amount')
    plot_feature_distributions(orange_samples_for_plotting, 'Avg Player Boost (Orange)',
                               feature_type='player', feature_idx=boost_idx,
                               wandb_log_name='orange_err_dist_player_boost')

    ball_vel_x_idx = GLOBAL_FEATURE_NAMES.index('ball_vel_x')
    plot_feature_distributions(orange_samples_for_plotting, 'Ball X Velocity (Orange)',
                               feature_type='global', feature_idx=ball_vel_x_idx,
                               wandb_log_name='orange_err_dist_ball_vel_x')

    seconds_remaining_idx = GLOBAL_FEATURE_NAMES.index('seconds_remaining')
    plot_feature_distributions(orange_samples_for_plotting, 'Seconds Remaining (Orange)',
                               feature_type='global', feature_idx=seconds_remaining_idx,
                               wandb_log_name='orange_err_dist_seconds_remaining')

    # Plot Average Positions for Orange Team Errors
    team_feature_idx = PLAYER_FEATURE_NAMES.index('p_team')
    if fn_orange_samples:
        plot_avg_positions(fn_orange_samples, 'FN Orange', team_feature_idx, 'orange_avg_pos_FN')
    if fp_orange_samples:
        plot_avg_positions(fp_orange_samples, 'FP Orange', team_feature_idx, 'orange_avg_pos_FP')
    if tp_orange_samples: # For reference
        plot_avg_positions(tp_orange_samples, 'TP Orange', team_feature_idx, 'orange_avg_pos_TP')

    # Log example misclassified samples to WandB Table
    log_misclassified_samples_table(fn_orange_samples, "Orange_FN", num_samples_to_log=10)
    log_misclassified_samples_table(fp_orange_samples, "Orange_FP", num_samples_to_log=10)

    # --- End of Error Analysis & Visualization ---

    if wandb.run:
        wandb.finish()
    print("[STATUS] Script execution finished.")


if __name__ == "__main__":
    # Set for debugging CUDA errors. Remove for production.
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 
    # torch.autograd.set_detect_anomaly(True) # For debugging gradient issues. Remove for production.
    main()