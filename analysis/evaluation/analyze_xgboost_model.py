import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# ====================== HELPER FUNCTIONS (CORRECTED) ======================
def load_data_as_df_for_xgb(list_of_csv_paths):
    """
    Loads data for XGBoost. CRUCIALLY, it does NOT normalize the features.
    """
    print(f"--- Loading data from {len(list_of_csv_paths)} files for XGBoost... ---")
    df_list = [pd.read_csv(path) for path in tqdm(list_of_csv_paths)]
    full_df = pd.concat(df_list, ignore_index=True).dropna()
    
    feature_cols = [col for col in full_df.columns if col.startswith('p') or col.startswith('ball') or col.startswith('boost') or col == 'seconds_remaining']
    X = full_df[feature_cols].copy()
    y_orange = full_df['team_1_goal_in_event_window']
    y_blue = full_df['team_0_goal_in_event_window']
    
    return X, y_orange, y_blue

def find_optimal_threshold(y_true, y_pred_proba):
    """Finds the optimal decision threshold from a validation set to maximize the F1-score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_f1_idx = np.argmax(f1_scores[:-1])
    return thresholds[best_f1_idx], f1_scores[best_f1_idx]

def plot_and_save_distribution(y_true, y_pred_proba, threshold, model_type, team_name, set_name, output_dir):
    plt.figure(figsize=(12, 8))
    sns.histplot(y_pred_proba[y_true == 0], bins=50, color='skyblue', label='Actual: No Goal', kde=True, log_scale=(False, True))
    sns.histplot(y_pred_proba[y_true == 1], bins=50, color='salmon', label='Actual: Goal', kde=True, log_scale=(False, True))
    plt.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Default Threshold (0.5)')
    plt.axvline(threshold, color='red', linestyle='-', linewidth=2, label=f'Optimal Threshold ({threshold:.4f})')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title(f'Log-Scale Prob. Distribution on {set_name} Set ({model_type.upper()} - {team_name.upper()})', fontsize=16)
    plt.xlabel('Predicted Probability of Goal'); plt.ylabel('Frequency (Log Scale)'); plt.legend()
    filepath = os.path.join(output_dir, f'{team_name}_{set_name.lower()}_distribution.png')
    plt.savefig(filepath); plt.close()
    print(f"  Distribution plot saved to: {filepath}")

def plot_and_save_confusion_matrix(y_true, preds, threshold, model_type, team_name, set_name, output_dir):
    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Goal", "Goal"], yticklabels=["No Goal", "Goal"])
    plt.title(f'CM on {set_name} Set ({model_type.upper()} - {team_name.upper()}) at Threshold {threshold:.4f}', fontsize=14)
    plt.ylabel('Actual Label'); plt.xlabel('Predicted Label');
    thresh_str = str(round(threshold, 4)).replace('.', 'p')
    filepath = os.path.join(output_dir, f'{team_name}_{set_name.lower()}_cm_thresh_{thresh_str}.png')
    plt.savefig(filepath, bbox_inches='tight'); plt.close()
    print(f"  Confusion matrix plot saved to: {filepath}")

def plot_and_save_metrics_table(metrics_default, metrics_optimized, threshold, model_type, team_name, set_name, output_dir):
    metrics_data = [[f"{metrics_default[0]:.4f}", f"{metrics_optimized[0]:.4f}"],
                    [f"{metrics_default[1]:.4f}", f"{metrics_optimized[1]:.4f}"],
                    [f"{metrics_default[2]:.4f}", f"{metrics_optimized[2]:.4f}"]]
    fig, ax = plt.subplots(figsize=(10, 2)); ax.axis('tight'); ax.axis('off')
    table = ax.table(cellText=metrics_data, rowLabels=['Precision', 'Recall', 'F1-Score'], colLabels=[f'Default (0.5)', f'Optimal ({threshold:.4f})'],
                     cellLoc='center', loc='center', colWidths=[0.3, 0.3])
    table.auto_set_font_size(False); table.set_fontsize(12); table.scale(1.2, 1.2)
    plt.title(f'Performance Metrics on {set_name} Set ({model_type.upper()} - {team_name.upper()})', fontsize=14, pad=20)
    filepath = os.path.join(output_dir, f'{team_name}_{set_name.lower()}_metrics_table.png')
    plt.savefig(filepath, bbox_inches='tight', dpi=150); plt.close()
    print(f"  Metrics table plot saved to: {filepath}")

# ====================== MAIN EXECUTION ======================
def main():
    parser = argparse.ArgumentParser(description="Full Analysis and Optimization for XGBoost Model")
    parser.add_argument('--model-path-orange', type=str, required=True, help='Path to the saved ORANGE XGBoost model (.json file).')
    parser.add_argument('--model-path-blue', type=str, required=True, help='Path to the saved BLUE XGBoost model (.json file).')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to the parent directory of train/val/test splits.')
    parser.add_argument('--output-dir', type=str, default='./analysis_results_xgb', help='Directory to save all output figures.')
    args = parser.parse_args()

    # --- 1. Load Data (Without Normalization) ---
    val_dir = os.path.join(args.data_dir, 'val'); test_dir = os.path.join(args.data_dir, 'test')
    val_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.csv')]
    test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.csv')]
    
    X_val, y_orange_val, y_blue_val = load_data_as_df_for_xgb(val_files)
    X_test, y_orange_test, y_blue_test = load_data_as_df_for_xgb(test_files)

    # --- 2. Load Models ---
    print(f"\n--- Loading ORANGE XGBoost model from {args.model_path_orange} ---")
    try:
        model_orange = xgb.XGBClassifier(); model_orange.load_model(args.model_path_orange)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load the ORANGE model. Error: {e}"); return
        
    print(f"--- Loading BLUE XGBoost model from {args.model_path_blue} ---")
    try:
        model_blue = xgb.XGBClassifier(); model_blue.load_model(args.model_path_blue)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load the BLUE model. Error: {e}"); return

    # --- 3. Find Optimal Thresholds on Validation Set ---
    print("\n--- Step 1: Finding Optimal Thresholds using VALIDATION set ---")
    val_probs_orange = model_orange.predict_proba(X_val)[:, 1]
    val_probs_blue = model_blue.predict_proba(X_val)[:, 1]
    
    optimal_threshold_orange, max_f1_val_orange = find_optimal_threshold(y_orange_val, val_probs_orange)
    optimal_threshold_blue, max_f1_val_blue = find_optimal_threshold(y_blue_val, val_probs_blue)
    
    print(f"  Optimal Threshold for Orange: {optimal_threshold_orange:.4f} (achieved F1: {max_f1_val_orange:.4f} on val set)")
    print(f"  Optimal Threshold for Blue:   {optimal_threshold_blue:.4f} (achieved F1: {max_f1_val_blue:.4f} on val set)")

    # --- 4. Evaluate on Test Set and Generate All Outputs ---
    print("\n--- Step 2: Evaluating on TEST set and Generating Figures ---")
    test_probs_orange = model_orange.predict_proba(X_test)[:, 1]
    test_probs_blue = model_blue.predict_proba(X_test)[:, 1]

    # --- Analysis for ORANGE team ---
    print("\n--- Analysis for ORANGE Team ---")
    team_name_o = 'orange'
    output_dir_o = os.path.join(args.output_dir, team_name_o)
    os.makedirs(output_dir_o, exist_ok=True)
    
    test_preds_def_o = (test_probs_orange > 0.5).astype(int)
    test_preds_opt_o = (test_probs_orange > optimal_threshold_orange).astype(int)
    metrics_def_o = [precision_score(y_orange_test, test_preds_def_o, zero_division=0), recall_score(y_orange_test, test_preds_def_o, zero_division=0), f1_score(y_orange_test, test_preds_def_o, zero_division=0)]
    metrics_opt_o = [precision_score(y_orange_test, test_preds_opt_o, zero_division=0), recall_score(y_orange_test, test_preds_opt_o, zero_division=0), f1_score(y_orange_test, test_preds_opt_o, zero_division=0)]
    
    plot_and_save_distribution(y_orange_test, test_probs_orange, optimal_threshold_orange, 'xgb', team_name_o, "Test", output_dir_o)
    plot_and_save_confusion_matrix(y_orange_test, test_preds_def_o, 0.5, 'xgb', team_name_o, "Test", output_dir_o)
    plot_and_save_confusion_matrix(y_orange_test, test_preds_opt_o, optimal_threshold_orange, 'xgb', team_name_o, "Test", output_dir_o)
    plot_and_save_metrics_table(metrics_def_o, metrics_opt_o, optimal_threshold_orange, 'xgb', team_name_o, "Test", output_dir_o)

    # --- Analysis for BLUE team ---
    print("\n--- Analysis for BLUE Team ---")
    team_name_b = 'blue'
    output_dir_b = os.path.join(args.output_dir, team_name_b)
    os.makedirs(output_dir_b, exist_ok=True)
    
    test_preds_def_b = (test_probs_blue > 0.5).astype(int)
    test_preds_opt_b = (test_probs_blue > optimal_threshold_blue).astype(int)
    metrics_def_b = [precision_score(y_blue_test, test_preds_def_b, zero_division=0), recall_score(y_blue_test, test_preds_def_b, zero_division=0), f1_score(y_blue_test, test_preds_def_b, zero_division=0)]
    metrics_opt_b = [precision_score(y_blue_test, test_preds_opt_b, zero_division=0), recall_score(y_blue_test, test_preds_opt_b, zero_division=0), f1_score(y_blue_test, test_preds_opt_b, zero_division=0)]

    plot_and_save_distribution(y_blue_test, test_probs_blue, optimal_threshold_blue, 'xgb', team_name_b, "Test", output_dir_b)
    plot_and_save_confusion_matrix(y_blue_test, test_preds_def_b, 0.5, 'xgb', team_name_b, "Test", output_dir_b)
    plot_and_save_confusion_matrix(y_blue_test, test_preds_opt_b, optimal_threshold_blue, 'xgb', team_name_b, "Test", output_dir_b)
    plot_and_save_metrics_table(metrics_def_b, metrics_opt_b, optimal_threshold_blue, 'xgb', team_name_b, "Test", output_dir_b)

    print("\n--- Analysis Complete ---")

if __name__ == '__main__':
    main()