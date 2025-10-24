import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description="Train, visualize, and interpret a Polynomial Logistic Regression model.")
    parser.add_argument('--data-dir', type=str, required=True, help='Path to the root of the split dataset.')
    parser.add_argument('--output-dir', type=str, default='./polynomial_analysis_results', help='Directory to save output plots.')
    parser.add_argument('--polynomial-degree', type=int, default=3, help='The degree of the polynomial to fit.')
    parser.add_argument('--team', type=str, required=True, choices=['blue', 'orange'], help='The team to train the model for.')
    return parser.parse_args()

def load_data(file_paths, team_label_col):
    print(f"--- Loading data from {len(file_paths)} files...")
    required_cols = ['ball_pos_y', team_label_col]
    df_list = [pd.read_csv(path, usecols=required_cols) for path in tqdm(file_paths, desc="Loading CSVs")]
    full_df = pd.concat(df_list, ignore_index=True).dropna()
    X = full_df[['ball_pos_y']].values
    y = full_df[team_label_col].values
    return X, y

def find_optimal_threshold(y_true, y_pred_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_f1_idx = np.argmax(f1_scores[:-1]); return thresholds[best_f1_idx], f1_scores[best_f1_idx]

def plot_learned_curve(model, scaler, poly, X_test, y_test, team_name, degree, output_dir):
    print("--- Generating visualization of the learned model curve...")
    x_range = np.linspace(X_test.min(), X_test.max(), 1000).reshape(-1, 1)
    x_range_poly = poly.transform(x_range)
    x_range_scaled = scaler.transform(x_range_poly)
    y_probs = model.predict_proba(x_range_scaled)[:, 1]
    fig, ax1 = plt.subplots(figsize=(15, 8)); color = 'tab:red'
    ax1.set_xlabel('Ball Y-Axis Position (Distance from Center Line)')
    ax1.set_ylabel('Predicted Probability of Goal', color=color)
    ax1.plot(x_range, y_probs, color=color, linewidth=3, label=f'Degree {degree} Polynomial Model')
    ax1.tick_params(axis='y', labelcolor=color); ax1.set_ylim(0, y_probs.max() * 1.2); ax1.grid(True, linestyle='--', alpha=0.6)
    ax2 = ax1.twinx(); color = 'tab:blue'
    ax2.set_ylabel('Frequency of Actual Goals', color=color)
    goal_positions = X_test[y_test == 1]
    sns.histplot(goal_positions, bins=100, ax=ax2, color=color, alpha=0.5, label='Actual Goal Occurrences')
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title(f'Model\'s Learned Goal Probability vs. Actual Goal Locations ({team_name.upper()} Team)', fontsize=16)
    fig.tight_layout()
    filepath = os.path.join(output_dir, f'{team_name}_degree_{degree}_probability_curve.png')
    plt.savefig(filepath, dpi=150); plt.close()
    print(f"  > Plot saved to: {filepath}")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    team_label_col = 'team_0_goal_in_event_window' if args.team == 'blue' else 'team_1_goal_in_event_window'
    
    train_files = [os.path.join(args.data_dir, 'train', f) for f in os.listdir(os.path.join(args.data_dir, 'train')) if f.endswith('.csv')]
    val_files = [os.path.join(args.data_dir, 'val', f) for f in os.listdir(os.path.join(args.data_dir, 'val')) if f.endswith('.csv')]
    test_files = [os.path.join(args.data_dir, 'test', f) for f in os.listdir(os.path.join(args.data_dir, 'test')) if f.endswith('.csv')]
    X_train, y_train = load_data(train_files, team_label_col)
    X_val, y_val = load_data(val_files, team_label_col)
    X_test, y_test = load_data(test_files, team_label_col)

    print(f"\n--- Training Polynomial Logistic Regression (Degree: {args.polynomial_degree}) for {args.team.upper()} team ---")
    poly = PolynomialFeatures(degree=args.polynomial_degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    model = LogisticRegression(class_weight='balanced', solver='saga', random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    print("--- Training complete ---")

    # =========================================================================
    # ===================== PRINTING WEIGHTS =====================
    # =========================================================================
    print("\n--- Model Coefficients (Learned Weights) ---")
    coefficients = model.coef_[0]
    feature_names = poly.get_feature_names_out(['ball_pos_y'])
    print("  The model learned the following importance for each feature:")
    for name, coef in zip(feature_names, coefficients):
        print(f"    - {name}: {coef:.4f}")
    print("\n  Interpretation:")
    print("  > A positive weight means increasing the feature value increases the probability of a goal.")
    print("  > The magnitude shows the feature's importance AFTER the data was scaled.")
    # =========================================================================

    print("\n--- Finding optimal threshold on validation set ---")
    X_val_poly = poly.transform(X_val)
    X_val_scaled = scaler.transform(X_val_poly)
    val_probs = model.predict_proba(X_val_scaled)[:, 1]
    optimal_thresh, _ = find_optimal_threshold(y_val, val_probs)
    print(f"  > Optimal decision threshold found: {optimal_thresh:.4f}")

    print("\n--- Evaluating performance on the unseen test set ---")
    X_test_poly = poly.transform(X_test)
    X_test_scaled = scaler.transform(X_test_poly)
    test_probs = model.predict_proba(X_test_scaled)[:, 1]
    test_preds_opt = (test_probs > optimal_thresh).astype(int)
    
    plot_learned_curve(model, scaler, poly, X_test, y_test, args.team, args.polynomial_degree, args.output_dir)

    f1 = f1_score(y_test, test_preds_opt); precision = precision_score(y_test, test_preds_opt); recall = recall_score(y_test, test_preds_opt)
    
    print("\n" + "="*50)
    print(f"  FINAL TEST RESULTS for {args.team.upper()} Team (Degree {args.polynomial_degree})")
    print("="*50)
    print(f"  Precision: {precision:.4f}"); print(f"  Recall:    {recall:.4f}"); print(f"  F1-Score:  {f1:.4f}")
    print("="*50)

if __name__ == '__main__':
    main()