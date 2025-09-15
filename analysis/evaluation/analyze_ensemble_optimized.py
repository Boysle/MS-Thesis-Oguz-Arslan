import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, confusion_matrix
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
def load_and_normalize_data(list_of_csv_paths):
    """Loads and NORMALIZES data for the MLP model."""
    print(f"--- Loading and NORMALIZING data from {len(list_of_csv_paths)} files for MLP... ---")
    df_list = [pd.read_csv(path) for path in tqdm(list_of_csv_paths)]
    full_df = pd.concat(df_list, ignore_index=True).dropna()
    
    all_features, y_orange, y_blue = [], [], []
    POS_MIN_X, POS_MAX_X = -4096, 4096; POS_MIN_Y, POS_MAX_Y = -6000, 6000; POS_MIN_Z, POS_MAX_Z = 0, 2044
    VEL_MIN, VEL_MAX = -2300, 2300; BOOST_MIN, BOOST_MAX = 0, 100; BALL_VEL_MIN, BALL_VEL_MAX = -6000, 6000
    BOOST_PAD_MIN, BOOST_PAD_MAX = 0, 10; DIST_MIN, DIST_MAX = 0, (8192**2 + 10240**2 + 2044**2)**0.5
    def normalize(val, min_val, max_val): return (val - min_val) / (max_val - min_val + 1e-8)
    
    for _, row in tqdm(full_df.iterrows(), total=len(full_df), desc="Normalizing rows"):
        try:
            player_features = [item for i in range(NUM_PLAYERS) for item in [normalize(float(row[f'p{i}_pos_x']), POS_MIN_X, POS_MAX_X), normalize(float(row[f'p{i}_pos_y']), POS_MIN_Y, POS_MAX_Y), normalize(float(row[f'p{i}_pos_z']), POS_MIN_Z, POS_MAX_Z), normalize(float(row[f'p{i}_vel_x']), VEL_MIN, VEL_MAX), normalize(float(row[f'p{i}_vel_y']), VEL_MIN, VEL_MAX), normalize(float(row[f'p{i}_vel_z']), VEL_MIN, VEL_MAX), float(row[f'p{i}_forward_x']), float(row[f'p{i}_forward_y']), float(row[f'p{i}_forward_z']), normalize(float(row[f'p{i}_boost_amount']), BOOST_MIN, BOOST_MAX), float(row[f'p{i}_team']), float(row[f'p{i}_alive']), normalize(float(row[f'p{i}_dist_to_ball']), DIST_MIN, DIST_MAX)]]
            global_features = [normalize(float(row['ball_pos_x']), POS_MIN_X, POS_MAX_X), normalize(float(row['ball_pos_y']), POS_MIN_Y, POS_MAX_Y), normalize(float(row['ball_pos_z']), POS_MIN_Z, POS_MAX_Z), normalize(float(row['ball_vel_x']), BALL_VEL_MIN, BALL_VEL_MAX), normalize(float(row['ball_vel_y']), BALL_VEL_MIN, BALL_VEL_MAX), normalize(float(row['ball_vel_z']), BALL_VEL_MIN, BALL_VEL_MAX), normalize(float(row['boost_pad_0_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX), normalize(float(row['boost_pad_1_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX), normalize(float(row['boost_pad_2_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX), normalize(float(row['boost_pad_3_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX), normalize(float(row['boost_pad_4_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX), normalize(float(row['boost_pad_5_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX), float(row['ball_hit_team_num']), normalize(min(float(row['seconds_remaining']), 300.0), 0, 300)]
            all_features.append(player_features + global_features)
            y_orange.append(int(float(row['team_1_goal_in_event_window'])))
            y_blue.append(int(float(row['team_0_goal_in_event_window'])))
        except (ValueError, KeyError): continue
    
    return np.array(all_features), pd.Series(y_orange), pd.Series(y_blue)

def load_raw_data_for_xgb(list_of_csv_paths):
    """Loads data for XGBoost. CRUCIALLY, it does NOT normalize the features."""
    print(f"--- Loading RAW data from {len(list_of_csv_paths)} files for XGBoost... ---")
    df_list = [pd.read_csv(path) for path in tqdm(list_of_csv_paths)]
    full_df = pd.concat(df_list, ignore_index=True).dropna()
    feature_cols = [col for col in full_df.columns if col.startswith('p') or col.startswith('ball') or col.startswith('boost') or col == 'seconds_remaining']
    return full_df[feature_cols], full_df['team_1_goal_in_event_window'], full_df['team_0_goal_in_event_window']

def find_optimal_threshold(y_true, y_pred_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    return thresholds[np.argmax(f1_scores[:-1])]

def plot_ensemble_distribution(y_true, y_pred_proba, team_name, output_dir):
    plt.figure(figsize=(12, 8)); sns.histplot(y_pred_proba[y_true == 0], bins=50, color='skyblue', label='Actual: No Goal', kde=True, log_scale=(False, True)); sns.histplot(y_pred_proba[y_true == 1], bins=50, color='salmon', label='Actual: Goal', kde=True, log_scale=(False, True)); plt.grid(True, which='both', linestyle='--', linewidth=0.5); plt.title(f'Ensemble Prob. Distribution on Test Set ({team_name.upper()})', fontsize=16); plt.xlabel('Predicted Probability of Goal'); plt.ylabel('Frequency (Log Scale)'); plt.legend(); filepath = os.path.join(output_dir, f'ensemble_{team_name}_distribution.png'); plt.savefig(filepath); plt.close(); print(f"  Ensemble distribution plot saved to: {filepath}")

def plot_ensemble_confusion_matrix(y_true, preds, team_name, output_dir):
    cm = confusion_matrix(y_true, preds); plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=["No Goal", "Goal"], yticklabels=["No Goal", "Goal"]); plt.title(f'Ensemble Confusion Matrix on Test Set ({team_name.upper()})', fontsize=14); plt.ylabel('Actual Label'); plt.xlabel('Predicted Label'); filepath = os.path.join(output_dir, f'ensemble_{team_name}_cm.png'); plt.savefig(filepath, bbox_inches='tight'); plt.close(); print(f"  Ensemble confusion matrix plot saved to: {filepath}")

def plot_final_comparison_table(metrics_mlp, metrics_xgb, metrics_ensemble, ensemble_optimized, team_name, output_dir):
    """Creates a final comparison table for all models, including the optimized ensemble."""
    metrics_data = [
        [f"{metrics_mlp[0]:.4f}", f"{metrics_xgb[0]:.4f}", f"{metrics_ensemble[0]:.4f}", f"{ensemble_optimized[0]:.4f}"],
        [f"{metrics_mlp[1]:.4f}", f"{metrics_xgb[1]:.4f}", f"{metrics_ensemble[1]:.4f}", f"{ensemble_optimized[1]:.4f}"],
        [f"{metrics_mlp[2]:.4f}", f"{metrics_xgb[2]:.4f}", f"{metrics_ensemble[2]:.4f}", f"{ensemble_optimized[2]:.4f}"]
    ]
    fig, ax = plt.subplots(figsize=(14, 2.5)); ax.axis('tight'); ax.axis('off')
    col_labels = ['Optimized MLP', 'Optimized XGB', 'Ensemble (Th=0.5)', 'Optimized Ensemble'] # <-- ADDED NEW COLUMN
    table = ax.table(cellText=metrics_data, rowLabels=['Precision', 'Recall', 'F1-Score'], 
                     colLabels=col_labels, cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(12); table.scale(1.2, 1.2)
    plt.title(f'Final Model Comparison on Test Set ({team_name.upper()})', fontsize=16, pad=20)
    filepath = os.path.join(output_dir, f'final_model_comparison_{team_name}.png')
    plt.savefig(filepath, bbox_inches='tight', dpi=150); plt.close()
    print(f"  Final comparison table saved to: {filepath}")

# ====================== MAIN ANALYSIS ======================
# ====================== MAIN ANALYSIS (FINAL VERSION) ======================
def main():
    parser = argparse.ArgumentParser(description="Analyze and Compare MLP, XGBoost, and Ensemble Models")
    parser.add_argument('--model-path-mlp', type=str, required=True, help='Path to the saved MLP model (.pth file).')
    parser.add_argument('--model-path-xgb-orange', type=str, required=True, help='Path to the saved ORANGE XGBoost model (.json file).')
    parser.add_argument('--model-path-xgb-blue', type=str, required=True, help='Path to the saved BLUE XGBoost model (.json file).')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to the parent directory of data splits.')
    parser.add_argument('--output-dir', type=str, default='./analysis_results_mlp_xgb_ensemble', help='Directory to save all output figures.')
    parser.add_argument('--batch-size', type=int, default=2048, help='Batch size for MLP evaluation.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Load Data ---
    val_dir = os.path.join(args.data_dir, 'val'); test_dir = os.path.join(args.data_dir, 'test')
    val_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.csv')]
    test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.csv')]
    
    # Data for MLP (Normalized)
    X_val_norm, y_orange_val, y_blue_val = load_and_normalize_data(val_files)
    X_test_norm, y_orange_test, y_blue_test = load_and_normalize_data(test_files)
    
    # Data for XGBoost (Raw)
    X_val_raw, _, _ = load_raw_data_for_xgb(val_files)
    X_test_raw, _, _ = load_raw_data_for_xgb(test_files)

    # --- 2. Load Models ---
    print(f"\n--- Loading Models ---")
    try:
        model_mlp = BaselineMLP().to(device)
        checkpoint = torch.load(args.model_path_mlp, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint: model_mlp.load_state_dict(checkpoint['model_state'])
        else: model_mlp.load_state_dict(checkpoint)
        model_mlp.eval(); print("MLP model loaded successfully.")
        
        model_xgb_orange = xgb.XGBClassifier(); model_xgb_orange.load_model(args.model_path_xgb_orange); print("XGBoost Orange model loaded successfully.")
        model_xgb_blue = xgb.XGBClassifier(); model_xgb_blue.load_model(args.model_path_xgb_blue); print("XGBoost Blue model loaded successfully.")
    except Exception as e: print(f"CRITICAL ERROR loading models: {e}"); return
            
    # --- 3. Get All Predictions ---
    print("\n--- Getting Predictions from All Models ---")
    with torch.no_grad():
        test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test_norm, dtype=torch.float32)); test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)
        mlp_probs_o, mlp_probs_b = [], []; val_mlp_probs_o, val_mlp_probs_b = [], []
        for (features,) in tqdm(test_loader, desc="Getting MLP Test predictions"):
            features = features.to(device); orange_prob, blue_prob = model_mlp(features)
            mlp_probs_o.extend(orange_prob.cpu().numpy().flatten()); mlp_probs_b.extend(blue_prob.cpu().numpy().flatten())
        mlp_probs_orange = np.array(mlp_probs_o); mlp_probs_blue = np.array(mlp_probs_b)
        
        val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val_norm, dtype=torch.float32)); val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)
        for (features,) in tqdm(val_loader, desc="Getting MLP Val predictions"):
            features = features.to(device); orange_prob, blue_prob = model_mlp(features)
            val_mlp_probs_o.extend(orange_prob.cpu().numpy().flatten()); val_mlp_probs_b.extend(blue_prob.cpu().numpy().flatten())
        val_mlp_probs_orange = np.array(val_mlp_probs_o); val_mlp_probs_blue = np.array(val_mlp_probs_b)

    xgb_probs_orange = model_xgb_orange.predict_proba(X_test_raw)[:, 1]; xgb_probs_blue = model_xgb_blue.predict_proba(X_test_raw)[:, 1]
    val_xgb_probs_orange = model_xgb_orange.predict_proba(X_val_raw)[:, 1]; val_xgb_probs_blue = model_xgb_blue.predict_proba(X_val_raw)[:, 1]

    # --- 4. Find Optimal Thresholds for ALL Models ---
    print("\n--- Finding Optimal Thresholds for All Models using VALIDATION set ---")
    optimal_threshold_mlp_orange = find_optimal_threshold(y_orange_val, val_mlp_probs_orange)
    optimal_threshold_xgb_orange = find_optimal_threshold(y_orange_val, val_xgb_probs_orange)
    ensemble_val_probs_orange = (val_mlp_probs_orange + val_xgb_probs_orange) / 2.0
    optimal_threshold_ensemble_orange = find_optimal_threshold(y_orange_val, ensemble_val_probs_orange)
    
    optimal_threshold_mlp_blue = find_optimal_threshold(y_blue_val, val_mlp_probs_blue)
    optimal_threshold_xgb_blue = find_optimal_threshold(y_blue_val, val_xgb_probs_blue)
    ensemble_val_probs_blue = (val_mlp_probs_blue + val_xgb_probs_blue) / 2.0
    optimal_threshold_ensemble_blue = find_optimal_threshold(y_blue_val, ensemble_val_probs_blue)

    print(f"  Optimal Thresholds (Orange): MLP={optimal_threshold_mlp_orange:.4f}, XGB={optimal_threshold_xgb_orange:.4f}, Ensemble={optimal_threshold_ensemble_orange:.4f}")
    print(f"  Optimal Thresholds (Blue):   MLP={optimal_threshold_mlp_blue:.4f}, XGB={optimal_threshold_xgb_blue:.4f}, Ensemble={optimal_threshold_ensemble_blue:.4f}")

    # --- 5. Evaluate All Models on TEST Set ---
    # ORANGE TEAM
    output_dir_o = os.path.join(args.output_dir, "orange"); os.makedirs(output_dir_o, exist_ok=True)
    print(f"\n--- Analysis for ORANGE Team (Results will be saved in {output_dir_o}) ---")
    mlp_preds_opt_o = (mlp_probs_orange > optimal_threshold_mlp_orange).astype(int)
    xgb_preds_opt_o = (xgb_probs_orange > optimal_threshold_xgb_orange).astype(int)
    ensemble_probs_o = (mlp_probs_orange + xgb_probs_orange) / 2.0
    ensemble_preds_o_default = (ensemble_probs_o > 0.5).astype(int)
    ensemble_preds_o_optimized = (ensemble_probs_o > optimal_threshold_ensemble_orange).astype(int)
    
    metrics_mlp_o = [precision_score(y_orange_test, mlp_preds_opt_o), recall_score(y_orange_test, mlp_preds_opt_o), f1_score(y_orange_test, mlp_preds_opt_o)]
    metrics_xgb_o = [precision_score(y_orange_test, xgb_preds_opt_o), recall_score(y_orange_test, xgb_preds_opt_o), f1_score(y_orange_test, xgb_preds_opt_o)]
    metrics_ensemble_o_default = [precision_score(y_orange_test, ensemble_preds_o_default), recall_score(y_orange_test, ensemble_preds_o_default), f1_score(y_orange_test, ensemble_preds_o_default)]
    metrics_ensemble_o_optimized = [precision_score(y_orange_test, ensemble_preds_o_optimized), recall_score(y_orange_test, ensemble_preds_o_optimized), f1_score(y_orange_test, ensemble_preds_o_optimized)]
    
    plot_ensemble_distribution(y_orange_test, ensemble_probs_o, "orange", output_dir_o)
    plot_ensemble_confusion_matrix(y_orange_test, ensemble_preds_o_optimized, "orange", output_dir_o)
    plot_final_comparison_table(metrics_mlp_o, metrics_xgb_o, metrics_ensemble_o_default, metrics_ensemble_o_optimized, "orange", output_dir_o)

    # BLUE TEAM
    output_dir_b = os.path.join(args.output_dir, "blue"); os.makedirs(output_dir_b, exist_ok=True)
    print(f"\n--- Analysis for BLUE Team (Results will be saved in {output_dir_b}) ---")
    mlp_preds_opt_b = (mlp_probs_blue > optimal_threshold_mlp_blue).astype(int)
    xgb_preds_opt_b = (xgb_probs_blue > optimal_threshold_xgb_blue).astype(int)
    ensemble_probs_b = (mlp_probs_blue + xgb_probs_blue) / 2.0
    ensemble_preds_b_default = (ensemble_probs_b > 0.5).astype(int)
    ensemble_preds_b_optimized = (ensemble_probs_b > optimal_threshold_ensemble_blue).astype(int)
    
    metrics_mlp_b = [precision_score(y_blue_test, mlp_preds_opt_b), recall_score(y_blue_test, mlp_preds_opt_b), f1_score(y_blue_test, mlp_preds_opt_b)]
    metrics_xgb_b = [precision_score(y_blue_test, xgb_preds_opt_b), recall_score(y_blue_test, xgb_preds_opt_b), f1_score(y_blue_test, xgb_preds_opt_b)]
    metrics_ensemble_b_default = [precision_score(y_blue_test, ensemble_preds_b_default), recall_score(y_blue_test, ensemble_preds_b_default), f1_score(y_blue_test, ensemble_preds_b_default)]
    metrics_ensemble_b_optimized = [precision_score(y_blue_test, ensemble_preds_b_optimized), recall_score(y_blue_test, ensemble_preds_b_optimized), f1_score(y_blue_test, ensemble_preds_b_optimized)]
    
    plot_ensemble_distribution(y_blue_test, ensemble_probs_b, "blue", output_dir_b)
    plot_ensemble_confusion_matrix(y_blue_test, ensemble_preds_b_optimized, "blue", output_dir_b)
    plot_final_comparison_table(metrics_mlp_b, metrics_xgb_b, metrics_ensemble_b_default, metrics_ensemble_b_optimized, "blue", output_dir_b)

    print("\n--- Analysis Complete ---")

if __name__ == '__main__':
    main()