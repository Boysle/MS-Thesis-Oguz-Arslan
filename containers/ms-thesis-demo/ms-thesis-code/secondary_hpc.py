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
        'team_0_goal_prev_5s', 'team_1_goal_prev_5s'
    ])
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")

    dataset = []

    for idx, row in df.iterrows():
        if row.isnull().any():
            continue
        if idx % 1000 == 0 and idx > 0:  # Print every 1000 samples
            print(f"[PROGRESS] Processed {idx}/{len(df)} samples")

        x = []
        for i in range(NUM_PLAYERS):
            pos_x = normalize(float(row[f'p{i}_pos_x']), POS_MIN_X, POS_MAX_X)
            pos_y = normalize(float(row[f'p{i}_pos_y']), POS_MIN_Y, POS_MAX_Y)
            pos_z = normalize(float(row[f'p{i}_pos_z']), POS_MIN_Z, POS_MAX_Z)
            vel_x = normalize(float(row[f'p{i}_vel_x']), VEL_MIN, VEL_MAX)
            vel_y = normalize(float(row[f'p{i}_vel_y']), VEL_MIN, VEL_MAX)
            vel_z = normalize(float(row[f'p{i}_vel_z']), VEL_MIN, VEL_MAX)
            forward_x = float(row[f'p{i}_forward_x'])
            forward_y = float(row[f'p{i}_forward_y'])
            forward_z = float(row[f'p{i}_forward_z'])
            boost = normalize(float(row[f'p{i}_boost_amount']), BOOST_MIN, BOOST_MAX)
            team = float(row[f'p{i}_team'])
            alive = float(row[f'p{i}_alive'])
            dist_to_ball = normalize(float(row[f'p{i}_dist_to_ball']), DIST_MIN, DIST_MAX)

            x.append([
                pos_x, pos_y, pos_z,
                vel_x, vel_y, vel_z,
                forward_x, forward_y, forward_z,
                boost, team, alive, dist_to_ball
            ])

        x = torch.tensor(x, dtype=torch.float32)

        # Check if the shape of x is correct
        assert x.shape == (NUM_PLAYERS, PLAYER_FEATURES), \
            f"Player features shape mismatch. Expected {(NUM_PLAYERS, PLAYER_FEATURES)}, got {x.shape}"

        edge_index = []
        edge_weights = []
        for i in range(NUM_PLAYERS):
            for j in range(NUM_PLAYERS):
                if i != j:
                    dist = torch.norm(x[i, :3] - x[j, :3])
                    weight = 1.0 / (1.0 + dist)
                    edge_weights.append(weight if not torch.isnan(weight) else 0.0)
                    edge_index.append([i, j])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float32)

        seconds_remaining = float(row['seconds_remaining'])
        normalized_seconds = normalize(min(seconds_remaining, 300.0), 0, 300)

        global_features = torch.tensor([
            normalize(float(row['ball_pos_x']), POS_MIN_X, POS_MAX_X),
            normalize(float(row['ball_pos_y']), POS_MIN_Y, POS_MAX_Y),
            normalize(float(row['ball_pos_z']), POS_MIN_Z, POS_MAX_Z),
            normalize(float(row['ball_vel_x']), BALL_VEL_MIN, BALL_VEL_MAX),
            normalize(float(row['ball_vel_y']), BALL_VEL_MIN, BALL_VEL_MAX),
            normalize(float(row['ball_vel_z']), BALL_VEL_MIN, BALL_VEL_MAX),
            normalize(float(row['boost_pad_0_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX),
            float(row['ball_hit_team_num']),
            normalized_seconds
        ], dtype=torch.float32)

        # Check if the shape of global_features is correct
        assert global_features.shape == (GLOBAL_FEATURE_DIM,), \
            f"Global features shape mismatch. Expected {(GLOBAL_FEATURE_DIM,)}, got {global_features.shape}"

        orange_y = torch.tensor([float(row['team_0_goal_prev_5s'])], dtype=torch.float32)
        blue_y = torch.tensor([float(row['team_1_goal_prev_5s'])], dtype=torch.float32)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weights,
            global_features=global_features.unsqueeze(0),
            orange_y=orange_y,
            blue_y=blue_y
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
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        x = global_mean_pool(x, data.batch)
        x = torch.cat([x, data.global_features], dim=1)
        return self.orange_head(x), self.blue_head(x)


# ====================== LOGGING ======================
def log_feature_importance_to_wandb(node_feature_grads, global_feature_grads, epoch):
    wandb.log({
        'epoch': epoch,
        'node_feature_importance': wandb.Histogram(node_feature_grads.cpu().numpy()),
        'global_feature_importance': wandb.Histogram(global_feature_grads.cpu().numpy()),
    })


# ====================== TRAINING ======================
def main():
    print("[STATUS] Starting script execution")
    args = parse_args()
    print(f"[CONFIG] Loaded configuration: {vars(args)}")

    # Create checkpoint directory if needed
    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)

    # Checkpoint initialization
    start_epoch = 0
    best_f1 = 0

    # Initialize with proper device mapping
    def load_checkpoint(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        return checkpoint['epoch'] + 1, checkpoint['best_f1']

    if args.resume and os.path.exists(args.checkpoint_path):
        start_epoch, best_f1 = load_checkpoint(args.checkpoint_path)
        print(f"[CHECKPOINT] Resuming from {args.checkpoint_path} at epoch {start_epoch}")
    else:
        start_epoch, best_f1 = 0, 0


    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Using device: {device}")
    if str(device) == 'cuda':
        print("[OPTIM] Enabling CUDA optimizations")
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')

    # WandB initialization
    try:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(project="rocket-league-gcn-safe", config=args)
        print("[LOGGING] Successfully initialized Weights & Biases")
    except Exception as e:
        print(f"[WARNING] Failed to initialize WandB: {str(e)}")

    dataset = load_and_process_data(args.csv_path)
    train_data, test_data = train_test_split(dataset, test_size=args.test_size, random_state=args.random_seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    model = SafeRocketLeagueGCN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCELoss()


    for epoch in range(start_epoch, args.epochs):
        print(f"\n[EPOCH] Starting epoch {epoch+1}/{args.epochs}")
        model.train()
        total_loss = 0
        correct_orange = 0
        correct_blue = 0
        all_orange_preds = []
        all_orange_labels = []
        all_blue_preds = []
        all_blue_labels = []

        # Training loop
        for batch in train_loader:
            batch = batch.to(device)
            batch.x.requires_grad_(True)
            batch.global_features.requires_grad_(True)
            optimizer.zero_grad()
            orange_pred, blue_pred = model(batch)
            loss_orange = criterion(orange_pred, batch.orange_y.unsqueeze(1))
            loss_blue = criterion(blue_pred, batch.blue_y.unsqueeze(1))
            loss = loss_orange + loss_blue
            loss.backward()
            optimizer.step()

            # Log gradients
            if batch.x.grad is not None and batch.global_features.grad is not None:
                node_feature_grads = batch.x.grad.abs().mean(dim=0)
                global_feature_grads = batch.global_features.grad.abs().mean(dim=0)
                log_feature_importance_to_wandb(node_feature_grads, global_feature_grads, epoch)
            else:
                print(f"Warning: No gradients detected for features in epoch {epoch}")

            total_loss += loss.item()
            pred_orange = (orange_pred > 0.5).float()
            pred_blue = (blue_pred > 0.5).float()

            correct_orange += (pred_orange == batch.orange_y.unsqueeze(1)).sum().item()
            correct_blue += (pred_blue == batch.blue_y.unsqueeze(1)).sum().item()

            all_orange_preds.extend(pred_orange.cpu().numpy())
            all_orange_labels.extend(batch.orange_y.cpu().numpy())
            all_blue_preds.extend(pred_blue.cpu().numpy())
            all_blue_labels.extend(batch.blue_y.cpu().numpy())

        orange_f1 = f1_score(all_orange_labels, all_orange_preds)
        blue_f1 = f1_score(all_blue_labels, all_blue_preds)
        orange_precision = precision_score(all_orange_labels, all_orange_preds)
        blue_precision = precision_score(all_blue_labels, all_blue_preds)
        orange_recall = recall_score(all_orange_labels, all_orange_preds)
        blue_recall = recall_score(all_blue_labels, all_blue_preds)
        orange_cm = confusion_matrix(all_orange_labels, all_orange_preds).tolist()
        blue_cm = confusion_matrix(all_blue_labels, all_blue_preds).tolist()

        wandb.log({
            'epoch': epoch,
            'train_loss': total_loss / len(train_loader),
            'train_acc_orange': correct_orange / len(train_data),
            'train_acc_blue': correct_blue / len(train_data),
            'orange_f1': orange_f1,
            'blue_f1': blue_f1,
            'orange_precision': orange_precision,
            'blue_precision': blue_precision,
            'orange_recall': orange_recall,
            'blue_recall': blue_recall,
        })

        # Flatten predictions and labels
        flat_orange_preds = [float(pred.item()) if isinstance(pred, torch.Tensor) else float(pred) for pred in all_orange_preds]
        flat_orange_labels = [int(label.item()) if isinstance(label, torch.Tensor) else int(label) for label in all_orange_labels]
        flat_blue_preds = [float(pred.item()) if isinstance(pred, torch.Tensor) else float(pred) for pred in all_blue_preds]
        flat_blue_labels = [int(label.item()) if isinstance(label, torch.Tensor) else int(label) for label in all_blue_labels]

        # Binarize predictions (e.g., threshold at 0.5)
        binary_orange_preds = [1 if pred >= 0.5 else 0 for pred in flat_orange_preds]
        binary_blue_preds = [1 if pred >= 0.5 else 0 for pred in flat_blue_preds]

        # Log to wandb
        wandb.log({
            'orange_confusion_matrix': wandb.plot.confusion_matrix(
                y_true=flat_orange_labels,
                preds=binary_orange_preds,
                class_names=["No Goal", "Goal"]
            ),
            'blue_confusion_matrix': wandb.plot.confusion_matrix(
                y_true=flat_blue_labels,
                preds=binary_blue_preds,
                class_names=["No Goal", "Goal"]
            )
        })

        # Print training stats
        print(f"Epoch {epoch:03d} | Loss: {total_loss/len(train_loader):.4f} | "
              f"Train Acc: O {correct_orange/len(train_data):.3f}, B {correct_blue/len(train_data):.3f} | "
              f"F1: O {orange_f1:.3f}, B {blue_f1:.3f} | "
              f"Precision: O {orange_precision:.3f}, B {blue_precision:.3f} | "
              f"Recall: O {orange_recall:.3f}, B {blue_recall:.3f}")
    
        model.eval()
        # Save checkpoint every N epochs
        if epoch % 2 == 0:  # Adjust frequency as needed
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_f1': best_f1,
                'args': vars(args)
            }, args.checkpoint_path)
            print(f"[CHECKPOINT] Saved at epoch {epoch}")

    test_orange_preds, test_orange_labels = [], []
    test_blue_preds, test_blue_labels = [], []

    # ======== TEST EVALUATION ========
    print("\n[TEST] Evaluating on test set")
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            orange_pred, blue_pred = model(batch)
            
            test_orange_preds.extend((orange_pred > 0.5).float().cpu().numpy())
            test_orange_labels.extend(batch.orange_y.cpu().numpy())
            test_blue_preds.extend((blue_pred > 0.5).float().cpu().numpy())
            test_blue_labels.extend(batch.blue_y.cpu().numpy())

    # Calculate metrics
    test_orange_f1 = f1_score(test_orange_labels, test_orange_preds)
    test_blue_f1 = f1_score(test_blue_labels, test_blue_preds)
    test_orange_cm = confusion_matrix(test_orange_labels, test_orange_preds)
    test_blue_cm = confusion_matrix(test_blue_labels, test_blue_preds)

    print(f"\n[TEST] Final Metrics:")
    print(f"Orange F1: {test_orange_f1:.4f} | Blue F1: {test_blue_f1:.4f}")
    print(f"Orange CM:\n{test_orange_cm}")
    print(f"Blue CM:\n{test_blue_cm}")

    # Log to WandB
    wandb.log({
        'test_orange_f1': test_orange_f1,
        'test_blue_f1': test_blue_f1,
        'test_orange_cm': wandb.plot.confusion_matrix(
            y_true=test_orange_labels, preds=test_orange_preds, 
            class_names=["No Goal", "Goal"]),
        'test_blue_cm': wandb.plot.confusion_matrix(
            y_true=test_blue_labels, preds=test_blue_preds,
            class_names=["No Goal", "Goal"])
    })
    # ======== END OF TEST EVALUATION ========

    wandb.finish()


if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    torch.autograd.set_detect_anomaly(True)
    main()
