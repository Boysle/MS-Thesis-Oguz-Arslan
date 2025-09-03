import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve

# ====================== CONFIGURATION & CONSTANTS (Must match your training script) ======================
NUM_PLAYERS = 6
PLAYER_FEATURES = 13
GLOBAL_FEATURES = 14 
TOTAL_FLAT_FEATURES = (NUM_PLAYERS * PLAYER_FEATURES) + GLOBAL_FEATURES

# Normalization bounds
POS_MIN_X, POS_MAX_X = -4096, 4096; POS_MIN_Y, POS_MAX_Y = -6000, 6000; POS_MIN_Z, POS_MAX_Z = 0, 2044
VEL_MIN, VEL_MAX = -2300, 2300; BOOST_MIN, BOOST_MAX = 0, 100; BALL_VEL_MIN, BALL_VEL_MAX = -6000, 6000
BOOST_PAD_MIN, BOOST_PAD_MAX = 0, 10; DIST_MIN, DIST_MAX = 0, (8192**2 + 10240**2 + 2044**2)**0.5

def normalize(val, min_val, max_val):
    return (val - min_val) / (max_val - min_val + 1e-8)

# ====================== MODEL & DATA DEFINITIONS (Must match your training script) ======================

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=TOTAL_FLAT_FEATURES):
        super(BaselineMLP, self).__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5)
        )
        self.orange_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.blue_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
    def forward(self, x):
        x_p = self.body(x)
        return self.orange_head(x_p), self.blue_head(x_p)

class EvaluationDataset(Dataset):
    def __init__(self, features, orange_labels, blue_labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.orange_labels = torch.tensor(orange_labels, dtype=torch.float32)
        self.blue_labels = torch.tensor(blue_labels, dtype=torch.float32)
    def __len__(self): return len(self.features)
    def __getitem__(self, idx): return self.features[idx], self.orange_labels[idx], self.blue_labels[idx]

def load_and_prepare_data(list_of_csv_paths):
    print(f"--- Loading data from {len(list_of_csv_paths)} files... ---")
    df_list = [pd.read_csv(path) for path in tqdm(list_of_csv_paths)]
    full_df = pd.concat(df_list, ignore_index=True).dropna()
    all_features, y_orange, y_blue = [], [], []
    for _, row in tqdm(full_df.iterrows(), total=len(full_df), desc="Processing rows"):
        player_features = [item for i in range(NUM_PLAYERS) for item in [normalize(float(row[f'p{i}_pos_x']), POS_MIN_X, POS_MAX_X), normalize(float(row[f'p{i}_pos_y']), POS_MIN_Y, POS_MAX_Y), normalize(float(row[f'p{i}_pos_z']), POS_MIN_Z, POS_MAX_Z), normalize(float(row[f'p{i}_vel_x']), VEL_MIN, VEL_MAX), normalize(float(row[f'p{i}_vel_y']), VEL_MIN, VEL_MAX), normalize(float(row[f'p{i}_vel_z']), VEL_MIN, VEL_MAX), float(row[f'p{i}_forward_x']), float(row[f'p{i}_forward_y']), float(row[f'p{i}_forward_z']), normalize(float(row[f'p{i}_boost_amount']), BOOST_MIN, BOOST_MAX), float(row[f'p{i}_team']), float(row[f'p{i}_alive']), normalize(float(row[f'p{i}_dist_to_ball']), DIST_MIN, DIST_MAX)]]
        global_features = [normalize(float(row['ball_pos_x']), POS_MIN_X, POS_MAX_X), normalize(float(row['ball_pos_y']), POS_MIN_Y, POS_MAX_Y), normalize(float(row['ball_pos_z']), POS_MIN_Z, POS_MAX_Z), normalize(float(row['ball_vel_x']), BALL_VEL_MIN, BALL_VEL_MAX), normalize(float(row['ball_vel_y']), BALL_VEL_MIN, BALL_VEL_MAX), normalize(float(row['ball_vel_z']), BALL_VEL_MIN, BALL_VEL_MAX), normalize(float(row['boost_pad_0_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX), normalize(float(row['boost_pad_1_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX), normalize(float(row['boost_pad_2_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX), normalize(float(row['boost_pad_3_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX), normalize(float(row['boost_pad_4_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX), normalize(float(row['boost_pad_5_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX), float(row['ball_hit_team_num']), normalize(min(float(row['seconds_remaining']), 300.0), 0, 300)]
        all_features.append(player_features + global_features)
        y_orange.append(int(float(row['team_1_goal_in_event_window'])))
        y_blue.append(int(float(row['team_0_goal_in_event_window'])))
    return np.array(all_features), np.array(y_orange), np.array(y_blue)

# ====================== HELPER FUNCTIONS ======================
def find_optimal_threshold(y_true, y_pred_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_f1_idx = np.argmax(f1_scores[:-1])
    return thresholds[best_f1_idx], f1_scores[best_f1_idx]

def get_predictions(model, dataloader, device):
    model.eval()
    all_orange_labels, all_blue_labels = [], []
    all_orange_probs, all_blue_probs = [], []
    with torch.no_grad():
        for features, orange_labels, blue_labels in tqdm(dataloader, desc="Getting predictions"):
            features = features.to(device)
            orange_pred_prob, blue_pred_prob = model(features)
            all_orange_labels.extend(orange_labels.numpy().flatten())
            all_blue_labels.extend(blue_labels.numpy().flatten())
            all_orange_probs.extend(orange_pred_prob.cpu().numpy().flatten())
            all_blue_probs.extend(blue_pred_prob.cpu().numpy().flatten())
    return np.array(all_orange_labels), np.array(all_blue_labels), np.array(all_orange_probs), np.array(all_blue_probs)

# ====================== MAIN EXECUTION ======================
def main():
    parser = argparse.ArgumentParser(description="Threshold Optimization for MLP Model")
    parser.add_argument('--model-path', type=str, required=True, help='Path to the saved MLP model checkpoint (.pth file).')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to the parent directory of train/val/test splits.')
    parser.add_argument('--batch-size', type=int, default=2048, help='Batch size for evaluation.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA evaluation.')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"--- Using device: {device} ---")

    # --- 1. Load Data ---
    val_dir = os.path.join(args.data_dir, 'val'); test_dir = os.path.join(args.data_dir, 'test')
    val_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.csv')]
    test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.csv')]
    X_val, y_orange_val, y_blue_val = load_and_prepare_data(val_files)
    X_test, y_orange_test, y_blue_test = load_and_prepare_data(test_files)
    val_dataset = EvaluationDataset(X_val, y_orange_val, y_blue_val)
    test_dataset = EvaluationDataset(X_test, y_orange_test, y_blue_test)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # --- 2. Load Model (CORRECTED) ---
    print(f"\n--- Loading model from {args.model_path} ---")
    try:
        model = BaselineMLP().to(device)
        # Load the entire checkpoint dictionary
        checkpoint = torch.load(args.model_path, map_location=device)
        # Extract the state_dict from the 'model_state' key
        model.load_state_dict(checkpoint['model_state'])
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load the model. Ensure the path is correct. Error: {e}")
        return

    # --- 3. Find Optimal Thresholds on Validation Set ---
    print("\n--- Step 1: Finding Optimal Thresholds using VALIDATION set ---")
    val_labels_orange, _, val_probs_orange, _ = get_predictions(model, val_loader, device)
    # Re-run for blue to keep it clean, though labels are the same
    _, val_labels_blue, _, val_probs_blue = get_predictions(model, val_loader, device)
    
    optimal_threshold_orange, max_f1_val_orange = find_optimal_threshold(val_labels_orange, val_probs_orange)
    optimal_threshold_blue, max_f1_val_blue = find_optimal_threshold(val_labels_blue, val_probs_blue)
    
    print(f"  Optimal Threshold for Orange: {optimal_threshold_orange:.4f} (achieved F1: {max_f1_val_orange:.4f} on val set)")
    print(f"  Optimal Threshold for Blue:   {optimal_threshold_blue:.4f} (achieved F1: {max_f1_val_blue:.4f} on val set)")

# --- 4. Evaluate on Test Set with Both Thresholds (CORRECTED) ---
print("\n--- Step 2: Evaluating on TEST set ---")
test_labels_orange, test_labels_blue, test_probs_orange, test_probs_blue = get_predictions(model, test_loader, device)

# A) Results with default 0.5 threshold
print("\n--- Results with DEFAULT 0.5 Threshold ---")
test_preds_default_orange = (test_probs_orange > 0.5).astype(int)
test_preds_default_blue = (test_probs_blue > 0.5).astype(int)
f1_def_o = f1_score(test_labels_orange, test_preds_default_orange, zero_division=0)
prec_def_o = precision_score(test_labels_orange, test_preds_default_orange, zero_division=0)
rec_def_o = recall_score(test_labels_orange, test_preds_default_orange, zero_division=0)
# Added "(on TEST set)" for clarity
print(f"  Default Orange (on TEST set) -> F1: {f1_def_o:.4f} | Precision: {prec_def_o:.4f} | Recall: {rec_def_o:.4f}")

f1_def_b = f1_score(test_labels_blue, test_preds_default_blue, zero_division=0)
prec_def_b = precision_score(test_labels_blue, test_preds_default_blue, zero_division=0)
rec_def_b = recall_score(test_labels_blue, test_preds_default_blue, zero_division=0)
print(f"  Default Blue   (on TEST set) -> F1: {f1_def_b:.4f} | Precision: {prec_def_b:.4f} | Recall: {rec_def_b:.4f}")

# B) Results with OPTIMIZED threshold
print("\n--- Results with OPTIMIZED Thresholds ---")
test_preds_optimized_orange = (test_probs_orange > optimal_threshold_orange).astype(int)
test_preds_optimized_blue = (test_probs_blue > optimal_threshold_blue).astype(int)

f1_opt_o = f1_score(test_labels_orange, test_preds_optimized_orange, zero_division=0)
prec_opt_o = precision_score(test_labels_orange, test_preds_optimized_orange, zero_division=0)
rec_opt_o = recall_score(test_labels_orange, test_preds_optimized_orange, zero_division=0)
# Added "(on TEST set)" for clarity
print(f"  Optimized Orange (on TEST set) -> F1: {f1_opt_o:.4f} | Precision: {prec_opt_o:.4f} | Recall: {rec_opt_o:.4f}")

f1_opt_b = f1_score(test_labels_blue, test_preds_optimized_blue, zero_division=0)
prec_opt_b = precision_score(test_labels_blue, test_preds_optimized_blue, zero_division=0)
rec_opt_b = recall_score(test_labels_blue, test_preds_optimized_blue, zero_division=0)
# Added "(on TEST set)" for clarity
print(f"  Optimized Blue   (on TEST set) -> F1: {f1_opt_b:.4f} | Precision: {prec_opt_b:.4f} | Recall: {rec_opt_b:.4f}")

print("\n--- Analysis Complete ---")

if __name__ == '__main__':
    main()