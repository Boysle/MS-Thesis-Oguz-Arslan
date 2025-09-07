import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# ====================== CONFIGURATION & DATA STRUCTURES ======================
NUM_PLAYERS = 6; PLAYER_FEATURES = 13; GLOBAL_FEATURES = 14 
TOTAL_FLAT_FEATURES = (NUM_PLAYERS * PLAYER_FEATURES) + GLOBAL_FEATURES

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=TOTAL_FLAT_FEATURES):
        super(BaselineMLP, self).__init__(); self.body = nn.Sequential(nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5)); self.orange_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid()); self.blue_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
    def forward(self, x): x_p = self.body(x); return self.orange_head(x_p), self.blue_head(x_p)

# ====================== HELPER FUNCTIONS ======================
def load_data_as_df(list_of_csv_paths, normalize_for_mlp=False):
    """Loads data and ONLY normalizes it if specifically requested for an MLP model."""
    print(f"--- Loading data from {len(list_of_csv_paths)} files... ---")
    df_list = [pd.read_csv(path) for path in tqdm(list_of_csv_paths)]
    full_df = pd.concat(df_list, ignore_index=True).dropna()
    
    feature_cols = [col for col in full_df.columns if col.startswith('p') or col.startswith('ball') or col.startswith('boost') or col == 'seconds_remaining']
    X = full_df[feature_cols].copy()

    if normalize_for_mlp:
        print("Normalizing features for MLP...")
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

def save_confusion_matrix_to_file(y_true, preds, threshold, team_name, model_type, filepath, set_name):
    """Formats a confusion matrix and appends it to a text file."""
    # ... (function is correct and remains unchanged, but we add set_name)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    header = f"Confusion Matrix for: {model_type.upper()} ({team_name.upper()}) on {set_name.upper()} SET at Threshold {threshold:.4f}\n"
    matrix_str = f"""
                  Predicted NEGATIVE | Predicted POSITIVE
Actual NEGATIVE:  {tn:<15,} | {fp:<15,}
Actual POSITIVE:  {fn:<15,} | {tp:<15,}
"""
    with open(filepath, 'a') as f: f.write(header); f.write("="*65 + "\n"); f.write(matrix_str); f.write("="*65 + "\n\n")

def plot_and_save_confusion_matrix(y_true, preds, threshold, team_name, model_type, filepath, set_name):
    # ... (function is correct and remains unchanged, but we add set_name)
    cm = confusion_matrix(y_true, preds); plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Goal", "Goal"], yticklabels=["No Goal", "Goal"])
    plt.title(f'CM: {model_type.upper()} ({team_name.upper()}) on {set_name.upper()} SET at Threshold {threshold:.4f}', fontsize=14)
    plt.ylabel('Actual Label'); plt.xlabel('Predicted Label')
    plt.savefig(filepath, bbox_inches='tight'); plt.close()
    print(f"Confusion matrix figure saved to: {filepath}")

def plot_and_save_metrics_table(metrics_data, threshold, team_name, model_type, filepath, set_name):
    # ... (function is correct and remains unchanged, but we add set_name)
    fig, ax = plt.subplots(figsize=(10, 2)); ax.axis('tight'); ax.axis('off')
    table = ax.table(cellText=metrics_data, rowLabels=['Precision', 'Recall', 'F1-Score'], colLabels=[f'Default (0.5)', f'Custom ({threshold:.4f})'],
                     cellLoc='center', loc='center', colWidths=[0.3, 0.3])
    table.auto_set_font_size(False); table.set_fontsize(12); table.scale(1.2, 1.2)
    plt.title(f'Performance Metrics on {set_name.upper()} SET ({model_type.upper()} - {team_name.upper()})', fontsize=14, pad=20)
    plt.savefig(filepath, bbox_inches='tight', dpi=150); plt.close()
    print(f"Metrics table figure saved to: {filepath}")

# ====================== MASTER ANALYSIS FUNCTION ======================
def run_full_analysis(model, model_type, X_data, y_data, team, threshold, set_name, batch_size, device):
    """
    Runs the complete analysis suite (predictions, histograms, tables, figures) for a given dataset.
    """
    print(f"\n{'='*25} Starting Analysis for {set_name.upper()} Set {'='*25}")

    # --- 1. Get Predictions ---
    if X_data is None or y_data is None or y_data.empty:
        print(f"--- No data found for {set_name} set. Skipping analysis. ---")
        return

    if model_type == 'mlp':
        model.eval()
        with torch.no_grad():
            dataset = torch.utils.data.TensorDataset(torch.tensor(X_data.values, dtype=torch.float32))
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
            all_probs = []
            for (features,) in tqdm(loader, desc=f"Getting MLP predictions for {set_name}"):
                features = features.to(device)
                orange_prob, blue_prob = model(features)
                probs_to_keep = orange_prob if team == 'orange' else blue_prob
                all_probs.extend(probs_to_keep.cpu().numpy().flatten())
            pred_proba = np.array(all_probs)
    else: # xgb
        pred_proba = model.predict_proba(X_data)[:, 1]

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

    # Text file for CMs
    cm_filepath_txt = f'./{model_type}_{team}_{set_name}_confusion_matrices.txt'
    if os.path.exists(cm_filepath_txt): os.remove(cm_filepath_txt)
    save_confusion_matrix_to_file(y_data, preds_default, 0.5, team, model_type, cm_filepath_txt, set_name)
    save_confusion_matrix_to_file(y_data, preds_optimized, threshold, team, model_type, cm_filepath_txt, set_name)
    print(f"Confusion matrices (text) saved to: {cm_filepath_txt}")
    
    # Figure for CMs
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

    # --- 1. Load ALL Data (Val and Test) ---
    val_dir = os.path.join(args.data_dir, 'val'); test_dir = os.path.join(args.data_dir, 'test')
    val_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.csv')]
    test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.csv')]
    
    should_normalize = (args.model_type == 'mlp')
    X_val, y_orange_val, y_blue_val = load_data_as_df(val_files, normalize_for_mlp=should_normalize)
    X_test, y_orange_test, y_blue_test = load_data_as_df(test_files, normalize_for_mlp=should_normalize)
    
    y_val = y_orange_val if args.team == 'orange' else y_blue_val
    y_test = y_orange_test if args.team == 'orange' else y_blue_test

    # --- 2. Load Model ---
    print(f"\n--- Loading {args.model_type.upper()} model for {args.team.upper()} team from {args.model_path} ---")
    model = None
    if args.model_type == 'mlp':
        try:
            model = BaselineMLP().to(device)
            checkpoint = torch.load(args.model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                model.load_state_dict(checkpoint['model_state'])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"CRITICAL ERROR loading MLP model: {e}"); return
    elif args.model_type == 'xgb':
        try:
            model = xgb.XGBClassifier(); model.load_model(args.model_path)
        except Exception as e:
            print(f"CRITICAL ERROR loading XGBoost model: {e}"); return
    
    if model is None: return

    # --- 3. Run the Full Analysis Suite on Both Datasets ---
    run_full_analysis(model, args.model_type, X_val, y_val, args.team, args.threshold, set_name="validation", batch_size=args.batch_size, device=device)
    run_full_analysis(model, args.model_type, X_test, y_test, args.team, args.threshold, set_name="test", batch_size=args.batch_size, device=device)

    print("\n--- Analysis Complete ---")

if __name__ == '__main__':
    main()