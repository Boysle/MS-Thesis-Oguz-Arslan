import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a simple multi-feature Logistic Regression model.")
    parser.add_argument('--data-dir', type=str, required=True, help='Path to the root of the split dataset.')
    parser.add_argument('--output-dir', type=str, default='./multivar_analysis_results', help='Directory to save output plots.')
    parser.add_argument('--team', type=str, required=True, choices=['blue', 'orange'], help='The team to train the model for.')
    return parser.parse_args()

def load_data(file_paths, feature_cols, team_label_col):
    """
    Efficiently loads only the necessary feature and label columns
    from a list of CSV files into a pandas DataFrame.
    """
    print(f"--- Loading data from {len(file_paths)} files...")
    required_cols = feature_cols + [team_label_col]
    
    df_list = [pd.read_csv(path, usecols=required_cols) for path in tqdm(file_paths, desc="Loading CSVs")]
    full_df = pd.concat(df_list, ignore_index=True).dropna()
    
    X = full_df[feature_cols].values
    y = full_df[team_label_col].values
    
    return X, y

def find_optimal_threshold(y_true, y_pred_proba):
    """Finds the optimal threshold to maximize F1 score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_f1_idx = np.argmax(f1_scores[:-1]); return thresholds[best_f1_idx], f1_scores[best_f1_idx]

def plot_feature_importance(coefficients, feature_names, team_name, output_dir):
    """
    Creates and saves a bar plot of the learned model coefficients (feature importance).
    """
    print("--- Generating feature importance plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=coefficients, y=feature_names, ax=ax, palette="vlag")
    
    ax.set_title(f'Feature Importance for {team_name.upper()} Goal Prediction', fontsize=16)
    ax.set_xlabel('Coefficient (Weight)', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.axvline(0, color='black', linewidth=0.8) # Add a line at zero for reference
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, f'{team_name}_feature_importance.png')
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  > Plot saved to: {filepath}")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    team_label_col = 'team_0_goal_in_event_window' if args.team == 'blue' else 'team_1_goal_in_event_window'
    feature_cols = ['ball_pos_y', 'ball_vel_y', 'ball_hit_team_num']
    
    # --- 1. Load Data for Train, Validation, and Test sets ---
    train_files = [os.path.join(args.data_dir, 'train', f) for f in os.listdir(os.path.join(args.data_dir, 'train')) if f.endswith('.csv')]
    val_files = [os.path.join(args.data_dir, 'val', f) for f in os.listdir(os.path.join(args.data_dir, 'val')) if f.endswith('.csv')]
    test_files = [os.path.join(args.data_dir, 'test', f) for f in os.listdir(os.path.join(args.data_dir, 'test')) if f.endswith('.csv')]

    X_train, y_train = load_data(train_files, feature_cols, team_label_col)
    X_val, y_val = load_data(val_files, feature_cols, team_label_col)
    X_test, y_test = load_data(test_files, feature_cols, team_label_col)

    # --- 2. Scale Features and Train the Model ---
    print(f"\n--- Training Logistic Regression for {args.team.upper()} team ---")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = LogisticRegression(class_weight='balanced', solver='saga', random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    print("--- Training complete ---")

    # --- 3. Display and Plot Feature Importance (Weights) ---
    coefficients = model.coef_[0]
    print("\n--- Model Coefficients (Learned Weights) ---")
    print("  The model learned the following importance for each feature:")
    for name, coef in zip(feature_cols, coefficients):
        print(f"    - {name}: {coef:.4f}")
    
    plot_feature_importance(coefficients, feature_cols, args.team, args.output_dir)

    # --- 4. Find Optimal Threshold on Validation Set ---
    print("\n--- Finding optimal threshold on validation set ---")
    X_val_scaled = scaler.transform(X_val)
    val_probs = model.predict_proba(X_val_scaled)[:, 1]
    optimal_thresh, _ = find_optimal_threshold(y_val, val_probs)
    print(f"  > Optimal decision threshold found: {optimal_thresh:.4f}")

    # --- 5. Evaluate on the Test Set ---
    print("\n--- Evaluating performance on the unseen test set ---")
    X_test_scaled = scaler.transform(X_test)
    test_probs = model.predict_proba(X_test_scaled)[:, 1]
    test_preds_opt = (test_probs > optimal_thresh).astype(int)
    
    f1 = f1_score(y_test, test_preds_opt); precision = precision_score(y_test, test_preds_opt); recall = recall_score(y_test, test_preds_opt)
    
    print("\n" + "="*50)
    print(f"  FINAL TEST RESULTS for {args.team.upper()} Team")
    print("="*50)
    print(f"  Precision: {precision:.4f}"); print(f"  Recall:    {recall:.4f}"); print(f"  F1-Score:  {f1:.4f}")
    print("="*50)

if __name__ == '__main__':
    main()