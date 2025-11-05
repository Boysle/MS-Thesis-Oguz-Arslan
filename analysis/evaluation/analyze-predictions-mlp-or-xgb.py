import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

##### NEW/MODIFIED #####
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, log_loss
from torch.utils.data import TensorDataset, DataLoader

# ====================== CONFIGURATION & DATA STRUCTURES ======================
NUM_PLAYERS = 6; PLAYER_FEATURES = 13; GLOBAL_FEATURES = 14 
TOTAL_FLAT_FEATURES = (NUM_PLAYERS * PLAYER_FEATURES) + GLOBAL_FEATURES

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=TOTAL_FLAT_FEATURES):
        super(BaselineMLP, self).__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5), 
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5)
        )
        
        ##### NEW/MODIFIED #####
        # Removed nn.Sigmoid() to match the trained models (which output logits)
        self.orange_head = nn.Sequential(nn.Linear(128, 1))
        self.blue_head = nn.Sequential(nn.Linear(128, 1))

    def forward(self, x): 
        x_p = self.body(x)
        return self.orange_head(x_p), self.blue_head(x_p)

# ====================== HELPER FUNCTIONS ======================
def load_data_as_df(list_of_csv_paths, normalize_for_mlp=False):
    """Loads data and ONLY normalizes it if specifically requested for an MLP model."""
    print(f"--- Loading data from {len(list_of_csv_paths)} files... ---")
    df_list = [pd.read_csv(path) for path in tqdm(list_of_csv_paths)]
    full_df = pd.concat(df_list, ignore_index=True).dropna()
    
    ##### NEW/MODIFIED #####
    # Define the EXACT feature order from the training script
    player_cols = [
        f'p{i}_{feat}' for i in range(NUM_PLAYERS) 
        for feat in [
            'pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 
            'forward_x', 'forward_y', 'forward_z', 'boost_amount', 
            'team', 'alive', 'dist_to_ball'
        ]
    ]
    
    global_cols = [
        'ball_pos_x', 'ball_pos_y', 'ball_pos_z', 
        'ball_vel_x', 'ball_vel_y', 'ball_vel_z',
        'boost_pad_0_respawn', 'boost_pad_1_respawn', 'boost_pad_2_respawn',
        'boost_pad_3_respawn', 'boost_pad_4_respawn', 'boost_pad_5_respawn',
        'ball_hit_team_num', 'seconds_remaining'
    ]
    
    # This is the correct 92-feature order
    feature_cols = player_cols + global_cols
    
    # Select columns in the correct order
    X = full_df[feature_cols].copy()
    ##### END NEW/MODIFIED #####

    if normalize_for_mlp:
        print("Normalizing features for MLP...")
        # (Your normalization constants and logic are correct)
        POS_MIN_X, POS_MAX_X = -4096, 4096; POS_MIN_Y, POS_MAX_Y = -6000, 6000; POS_MIN_Z, POS_MAX_Z = 0, 2044
        VEL_MIN, VEL_MAX = -2300, 2300; BOOST_MIN, BOOST_MAX = 0, 100; BALL_VEL_MIN, BALL_VEL_MAX = -6000, 6000
        BOOST_PAD_MIN, BOOST_PAD_MAX = 0, 10; DIST_MIN, DIST_MAX = 0, (8192**2 + 10240**2 + 2044**2)**0.5
        def normalize(val, min_val, max_val): return (val - min_val) / (max_val - min_val + 1e-8)
        
        for col in tqdm(X.columns):
            if 'pos_x' in col or 'ball_pos_x' in col: X[col] = normalize(X[col], POS_MIN_X, POS_MAX_X)
            elif 'pos_y' in col or 'ball_pos_y' in col: X[col] = normalize(X[col], POS_MIN_Y, POS_MAX_Y)
            elif 'pos_z' in col or 'ball_pos_z' in col: X[col] = normalize(X[col], POS_MIN_Z, POS_MAX_Z)
            elif 'vel' in col and 'ball' not in col: X[col] = normalize(X[col], VEL_MIN, VEL_MAX)
            elif 'ball_vel' in col: X[col] = normalize(X[col], BALL_VEL_MIN, BALL_VEL_MAX)
            elif 'boost_amount' in col: X[col] = normalize(X[col], BOOST_MIN, BOOST_MAX)
            elif 'dist_to_ball' in col: X[col] = normalize(X[col], DIST_MIN, DIST_MAX)
            elif 'boost_pad' in col: X[col] = normalize(X[col], BOOST_PAD_MIN, BOOST_PAD_MAX)
            elif 'seconds_remaining' in col: X[col] = normalize(np.minimum(X[col], 300.0), 0, 300)

    return X, full_df['team_1_goal_in_event_window'], full_df['team_0_goal_in_event_window']

# (Other helper functions like save_confusion_matrix_to_file, plot_and_save... remain unchanged)
def save_confusion_matrix_to_file(y_true, preds, threshold, team_name, model_type, filepath, set_name):
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0,1]).ravel()
    header = f"Confusion Matrix for: {model_type.upper()} ({team_name.upper()}) on {set_name.upper()} SET at Threshold {threshold:.4f}\n"
    matrix_str = f"""
                  Predicted NEGATIVE | Predicted POSITIVE
Actual NEGATIVE:  {tn:<15,} | {fp:<15,}
Actual POSITIVE:  {fn:<15,} | {tp:<15,}
"""
    with open(filepath, 'a') as f: f.write(header); f.write("="*65 + "\n"); f.write(matrix_str); f.write("="*65 + "\n\n")

def plot_and_save_confusion_matrix(y_true, preds, threshold, team_name, model_type, filepath, set_name):
    cm = confusion_matrix(y_true, preds, labels=[0,1]); plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Goal", "Goal"], yticklabels=["No Goal", "Goal"])
    plt.title(f'CM: {model_type.upper()} ({team_name.upper()}) on {set_name.upper()} SET at Threshold {threshold:.4f}', fontsize=14)
    plt.ylabel('Actual Label'); plt.xlabel('Predicted Label')
    plt.savefig(filepath, bbox_inches='tight'); plt.close()
    print(f"Confusion matrix figure saved to: {filepath}")

def plot_and_save_metrics_table(metrics_data, threshold, team_name, model_type, filepath, set_name):
    fig, ax = plt.subplots(figsize=(10, 2)); ax.axis('tight'); ax.axis('off')
    table = ax.table(cellText=metrics_data, rowLabels=['Precision', 'Recall', 'F1-Score'], colLabels=[f'Default (0.5)', f'Custom ({threshold:.4f})'],
                     cellLoc='center', loc='center', colWidths=[0.3, 0.3])
    table.auto_set_font_size(False); table.set_fontsize(12); table.scale(1.2, 1.2)
    plt.title(f'Performance Metrics on {set_name.upper()} SET ({model_type.upper()} - {team_name.upper()})', fontsize=14, pad=20)
    plt.savefig(filepath, bbox_inches='tight', dpi=150); plt.close()
    print(f"Metrics table figure saved to: {filepath}")


# ====================== MASTER ANALYSIS FUNCTION ======================
##### NEW/MODIFIED #####
def run_full_analysis(model, model_type, X_data, y_orange_data, y_blue_data, team, threshold, set_name, batch_size, device, pos_weight_orange, pos_weight_blue):
    """
    Runs the complete analysis suite (predictions, loss, histograms, tables, figures) for a given dataset.
    """
    print(f"\n{'='*25} Starting Analysis for {set_name.upper()} Set {'='*25}")

    # Select the correct Y data for this team (used for visuals and XGB)
    y_data = y_orange_data if team == 'orange' else y_blue_data
    
    if X_data is None or y_data is None or y_data.empty:
        print(f"--- No data found for {set_name} set. Skipping analysis. ---")
        return

    # --- 1. Get Predictions & Calculate Loss ---
    if model_type == 'mlp':
        model.eval()
        
        # Create criteria with the loaded weights
        criterion_o = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_orange]).to(device))
        criterion_b = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_blue]).to(device))
        
        # Create a DataLoader that includes labels for loss calculation
        y_orange_tensor = torch.tensor(y_orange_data.values, dtype=torch.float32).view(-1, 1)
        y_blue_tensor = torch.tensor(y_blue_data.values, dtype=torch.float32).view(-1, 1)
        dataset = TensorDataset(torch.tensor(X_data.values, dtype=torch.float32), y_orange_tensor, y_blue_tensor)
        loader = DataLoader(dataset, batch_size=batch_size)
        
        all_probs = []
        total_loss_o, total_loss_b = 0.0, 0.0
        
        with torch.no_grad():
            for (features, y_o_batch, y_b_batch) in tqdm(loader, desc=f"Getting MLP predictions/loss for {set_name}"):
                features, y_o_batch, y_b_batch = features.to(device), y_o_batch.to(device), y_b_batch.to(device)
                
                # Get Logits
                orange_logits, blue_logits = model(features)
                
                # Calculate Loss from Logits
                total_loss_o += criterion_o(orange_logits, y_o_batch).item()
                total_loss_b += criterion_b(blue_logits, y_b_batch).item()

                # Get Probs for visuals
                orange_prob = torch.sigmoid(orange_logits)
                blue_prob = torch.sigmoid(blue_logits)
                
                probs_to_keep = orange_prob if team == 'orange' else blue_prob
                all_probs.extend(probs_to_keep.cpu().numpy().flatten())
        
        pred_proba = np.array(all_probs)
        
        # Finalize and print loss
        avg_loss_o = total_loss_o / len(loader)
        avg_loss_b = total_loss_b / len(loader)
        print(f"\n--- {set_name.upper()} SET TEST LOSS (MLP) ---")
        print(f"  Orange Loss: {avg_loss_o:.4f}")
        print(f"  Blue Loss:   {avg_loss_b:.4f}")
        print(f"  Total Loss:  {avg_loss_o + avg_loss_b:.4f}")

    else: # xgb
        pred_proba = model.predict_proba(X_data)[:, 1]
        
        # Calculate weighted log loss
        pos_weight_for_team = pos_weight_orange if team == 'orange' else pos_weight_blue
        sample_weights = np.where(y_data == 1, pos_weight_for_team, 1.0)
        test_loss = log_loss(y_data, pred_proba, sample_weight=sample_weights)
        
        print(f"\n--- {set_name.upper()} SET TEST LOSS (XGB) ---")
        print(f"  {team.title()} Loss: {test_loss:.4f}")

    # --- 2. Generate Probability Distribution Histogram ---
    histogram_path = f'./{model_type}_{team}_{set_name}_probability_distribution.png'
    plt.figure(figsize=(12, 8))
    sns.histplot(pred_proba[y_data == 0], bins=50, color='skyblue', label='Actual: No Goal', kde=True, log_scale=(False, True))
    sns.histplot(pred_proba[y_data == 1], bins=50, color='salmon', label='Actual: Goal', kde=True, log_scale=(False, True))
    plt.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Default Threshold (0.5)')
    plt.axvline(threshold, color='red', linestyle='-', linewidth=2, label=f'Custom Threshold ({threshold:.4f})')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title(f'Log-Scale Prob. Distribution on {set_name.upper()} SET ({model_type.upper()} - {team.upper()})', fontsize=16)
    plt.xlabel('Predicted Probability of Goal'); plt.ylabel('Frequency (Log Scale)'); plt.legend()
    plt.savefig(histogram_path); plt.close()
    print(f"\n[OUTPUT] Probability distribution histogram saved to: {histogram_path}")

    # --- 3. Generate Confusion Matrices and Tables ---
    preds_default = (pred_proba > 0.5).astype(int)
    preds_optimized = (pred_proba > threshold).astype(int)

    cm_filepath_txt = f'./{model_type}_{team}_{set_name}_confusion_matrices.txt'
    if os.path.exists(cm_filepath_txt): os.remove(cm_filepath_txt) # Clear old file
    save_confusion_matrix_to_file(y_data, preds_default, 0.5, team, model_type, cm_filepath_txt, set_name)
    save_confusion_matrix_to_file(y_data, preds_optimized, threshold, team, model_type, cm_filepath_txt, set_name)
    print(f"Confusion matrices (text) saved to: {cm_filepath_txt}")
    
    cm_fig_default_path = f'./{model_type}_{team}_{set_name}_cm_default.png'
    cm_fig_optimized_path = f'./{model_type}_{team}_{set_name}_cm_optimized.png'
    plot_and_save_confusion_matrix(y_data, preds_default, 0.5, team, model_type, cm_fig_default_path, set_name)
    plot_and_save_confusion_matrix(y_data, preds_optimized, threshold, team, model_type, cm_fig_optimized_path, set_name)

    # --- 4. Generate Metrics Table Figure ---
    f1_def = f1_score(y_data, preds_default, zero_division=0); prec_def = precision_score(y_data, preds_default, zero_division=0); rec_def = recall_score(y_data, preds_default, zero_division=0)
    f1_opt = f1_score(y_data, preds_optimized, zero_division=0); prec_opt = precision_score(y_data, preds_optimized, zero_division=0); rec_opt = recall_score(y_data, preds_optimized, zero_division=0)
    metrics_data = [[f"{prec_def:.4f}", f"{prec_opt:.4f}"], [f"{rec_def:.4f}", f"{rec_opt:.4f}"], [f"{f1_def:.4f}", f"{f1_opt:.4f}"]]
    table_filepath = f'./{model_type}_{team}_{set_name}_metrics_table.png'
    plot_and_save_metrics_table(metrics_data, threshold, team, model_type, table_filepath, set_name)

# ====================== MAIN ANALYSIS ======================
def main():
    parser = argparse.ArgumentParser(description="Analyze Model Predictions: Thresholds & Distributions")
    parser.add_argument('--model-type', type=str, required=True, choices=['mlp', 'xgb'], help='Type of the model to analyze.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the saved model file.')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to the parent directory of data splits.')
    parser.add_argument('--team', type=str, default='orange', choices=['orange', 'blue'], help='Which team model to analyze.')
    parser.add_argument('--threshold', type=float, required=True, help='The custom threshold to use for analysis.')
    parser.add_argument('--batch-size', type=int, default=2048, help='Batch size for MLP evaluation.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##### NEW/MODIFIED #####
    # --- 1. Calculate Class Weights (Needed for Loss) ---
    print("\n--- Calculating class weights from training data ---")
    train_dir = os.path.join(args.data_dir, 'train')
    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.csv')]
    if not train_files:
        print(f"CRITICAL ERROR: No training files found in {train_dir}. Cannot calculate weights."); return

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
    ##### END NEW/MODIFIED #####


    # --- 2. Load ALL Data (Val and Test) ---
    val_dir = os.path.join(args.data_dir, 'val'); test_dir = os.path.join(args.data_dir, 'test')
    val_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.csv')]
    test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.csv')]
    
    should_normalize = (args.model_type == 'mlp')
    X_val, y_orange_val, y_blue_val = load_data_as_df(val_files, normalize_for_mlp=should_normalize)
    X_test, y_orange_test, y_blue_test = load_data_as_df(test_files, normalize_for_mlp=should_normalize)
    

    # --- 3. Load Model ---
    print(f"\n--- Loading {args.model_type.upper()} model for {args.team.upper()} team from {args.model_path} ---")
    model = None
    if args.model_type == 'mlp':
        try:
            # Use the corrected BaselineMLP definition (no sigmoid)
            model = BaselineMLP().to(device)
            checkpoint = torch.load(args.model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                model.load_state_dict(checkpoint['model_state'])
            else:
                model.load_state_dict(checkpoint) # For older, raw state_dict saves
        except Exception as e:
            print(f"CRITICAL ERROR loading MLP model: {e}"); return
    elif args.model_type == 'xgb':
        try:
            model = xgb.XGBClassifier(); model.load_model(args.model_path)
        except Exception as e:
            print(f"CRITICAL ERROR loading XGBoost model: {e}"); return
    
    if model is None: return

    # --- 4. Run the Full Analysis Suite on Both Datasets ---
    ##### NEW/MODIFIED #####
    # Pass all data and weights to the analysis function
    run_full_analysis(model, args.model_type, X_val, y_orange_val, y_blue_val, args.team, args.threshold, 
                      set_name="validation", batch_size=args.batch_size, device=device, 
                      pos_weight_orange=pos_weight_orange, pos_weight_blue=pos_weight_blue)
    
    run_full_analysis(model, args.model_type, X_test, y_orange_test, y_blue_test, args.team, args.threshold, 
                      set_name="test", batch_size=args.batch_size, device=device, 
                      pos_weight_orange=pos_weight_orange, pos_weight_blue=pos_weight_blue)

    print("\n--- Analysis Complete ---")

if __name__ == '__main__':
    main()