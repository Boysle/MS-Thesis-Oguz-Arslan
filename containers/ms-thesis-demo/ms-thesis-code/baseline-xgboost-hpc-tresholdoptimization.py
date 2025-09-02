import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import linecache
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve
from xgboost import XGBClassifier
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

# ====================== CONFIGURATION & CONSTANTS ======================
NUM_PLAYERS = 6; PLAYER_FEATURES = 13; GLOBAL_FEATURES = 14
POS_MIN_X, POS_MAX_X = -4096, 4096; POS_MIN_Y, POS_MAX_Y = -6000, 6000; POS_MIN_Z, POS_MAX_Z = 0, 2044
VEL_MIN, VEL_MAX = -2300, 2300; BOOST_MIN, BOOST_MAX = 0, 100; BALL_VEL_MIN, BALL_VEL_MAX = -6000, 6000
BOOST_PAD_MIN, BOOST_PAD_MAX = 0, 10; DIST_MIN, DIST_MAX = 0, (8192**2 + 10240**2 + 2044**2)**0.5

def normalize(val, min_val, max_val):
    return (val - min_val) / (max_val - min_val + 1e-8)

# ====================== ARGUMENT PARSER ======================
def parse_args():
    parser = argparse.ArgumentParser(description="Rocket League Dual XGBoost Training (Fixed W&B plots)")
    parser.add_argument("--data-dir", type=str, default=r"E:\\Raw RL Esports Replays\\Day 3 Swiss Stage\\Round 1\\split_dataset", help="Path to dataset root")
    parser.add_argument('--wandb-project', type=str, default="rl-goal-prediction-baselines", help="W&B project name.")
    parser.add_argument('--run-name', type=str, default=None, help="Custom name for the W&B run.")
    parser.add_argument('--max-depth', type=int, default=6, help='XGBoost max depth.')
    parser.add_argument('--n-estimators', type=int, default=1000, help='Maximum number of trees.')
    parser.add_argument('--learning-rate', type=float, default=0.05, help='XGBoost learning rate.')
    parser.add_argument('--early-stopping-rounds', type=int, default=50, help='Early stopping patience.')
    parser.add_argument("--num-threads", type=int, default=-1, help="Number of threads. -1 uses all cores.")
    parser.add_argument('--model-save-path', type=str, default=None, help='Path to save the final models. Saves orange_model.json and blue_model.json.')
    return parser.parse_args()

# ====================== DATA LOADER ======================
class RobustLazyLoader:
    def __init__(self, list_of_csv_paths):
        self.csv_paths = list_of_csv_paths
        self.file_info, self.cumulative_rows, self.header, total_rows = [], [0], None, 0
        desc = f"Indexing {os.path.basename(os.path.dirname(list_of_csv_paths[0]))} files"
        for path in tqdm(self.csv_paths, desc=desc):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    if self.header is None: self.header = f.readline().strip().split(',')
                    num_lines = sum(1 for _ in f)
                if num_lines > 0:
                    self.file_info.append({'path': path, 'rows': num_lines})
                    total_rows += num_lines; self.cumulative_rows.append(total_rows)
            except Exception as e: 
                print(f"\nWarning: Could not process file {path}. Skipping. Error: {e}")
        self.length = total_rows

    def __len__(self): return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length: return None
        file_index = np.searchsorted(self.cumulative_rows, idx, side='right') - 1
        file_path = self.file_info[file_index]['path']
        local_idx = idx - self.cumulative_rows[file_index]
        line = linecache.getline(file_path, local_idx + 2)
        if not line: return None
        row = dict(zip(self.header, line.strip().split(',')))
        try:
            player_features = [item for i in range(NUM_PLAYERS) for item in [
                normalize(float(row[f'p{i}_pos_x']), POS_MIN_X, POS_MAX_X),
                normalize(float(row[f'p{i}_pos_y']), POS_MIN_Y, POS_MAX_Y),
                normalize(float(row[f'p{i}_pos_z']), POS_MIN_Z, POS_MAX_Z),
                normalize(float(row[f'p{i}_vel_x']), VEL_MIN, VEL_MAX),
                normalize(float(row[f'p{i}_vel_y']), VEL_MIN, VEL_MAX),
                normalize(float(row[f'p{i}_vel_z']), VEL_MIN, VEL_MAX),
                float(row[f'p{i}_forward_x']), float(row[f'p{i}_forward_y']), float(row[f'p{i}_forward_z']),
                normalize(float(row[f'p{i}_boost_amount']), BOOST_MIN, BOOST_MAX),
                float(row[f'p{i}_team']), float(row[f'p{i}_alive']),
                normalize(float(row[f'p{i}_dist_to_ball']), DIST_MIN, DIST_MAX)
            ]]
            global_features = [
                normalize(float(row['ball_pos_x']), POS_MIN_X, POS_MAX_X),
                normalize(float(row['ball_pos_y']), POS_MIN_Y, POS_MAX_Y),
                normalize(float(row['ball_pos_z']), POS_MIN_Z, POS_MAX_Z),
                normalize(float(row['ball_vel_x']), BALL_VEL_MIN, BALL_VEL_MAX),
                normalize(float(row['ball_vel_y']), BALL_VEL_MIN, BALL_VEL_MAX),
                normalize(float(row['ball_vel_z']), BALL_VEL_MIN, BALL_VEL_MAX),
                normalize(float(row['boost_pad_0_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX),
                normalize(float(row['boost_pad_1_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX),
                normalize(float(row['boost_pad_2_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX),
                normalize(float(row['boost_pad_3_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX),
                normalize(float(row['boost_pad_4_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX),
                normalize(float(row['boost_pad_5_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX),
                float(row['ball_hit_team_num']),
                normalize(min(float(row['seconds_remaining']), 300.0), 0, 300)
            ]
            features = np.array(player_features + global_features, dtype=np.float32)
            orange_y = int(float(row['team_1_goal_in_event_window']))
            blue_y = int(float(row['team_0_goal_in_event_window']))
            return features, orange_y, blue_y
        except: 
            return None

    def to_dataframe(self):
        desc = f"Loading {os.path.basename(os.path.dirname(self.csv_paths[0]))} data into memory"
        print(f"--- {desc}... ---")
        df_list = [pd.read_csv(info['path']) for info in tqdm(self.file_info, desc=desc)]
        if not df_list: return None, None, None
        full_df = pd.concat(df_list, ignore_index=True).dropna()
        if full_df.empty: return None, None, None
        
        feature_cols = [col for col in full_df.columns if col.startswith('p') or col.startswith('ball') or col.startswith('boost') or col == 'seconds_remaining']
        X = full_df[feature_cols]
        y_orange = full_df['team_1_goal_in_event_window']
        y_blue = full_df['team_0_goal_in_event_window']
        return X, y_orange, y_blue

# ====================== EVALUATION ======================
def evaluate_and_log(model, X_data, y_true, team_name, set_name="val"):
    y_true_np = y_true.to_numpy().flatten()
    if y_true_np.size == 0:
        print(f"  WARNING: No data to evaluate for {team_name} on {set_name} set. Skipping.")
        return

    y_pred_proba_flat = model.predict_proba(X_data)[:, 1].flatten()
    y_pred_binary_flat = (y_pred_proba_flat > 0.5).astype(int)

    f1 = f1_score(y_true_np, y_pred_binary_flat, zero_division=0)
    precision = precision_score(y_true_np, y_pred_binary_flat, zero_division=0)
    recall = recall_score(y_true_np, y_pred_binary_flat, zero_division=0)
    print(f"  {set_name.upper()} {team_name} -> F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

    if wandb.run:
        y_true_list = y_true_np.tolist()
        y_pred_list = y_pred_binary_flat.tolist()
        y_probas_for_plots = np.stack([1 - y_pred_proba_flat, y_pred_proba_flat], axis=1)

        log_dict = {
            f"{set_name}/f1_{team_name}": f1,
            f"{set_name}/precision_{team_name}": precision,
            f"{set_name}/recall_{team_name}": recall,
        }

        try:
            log_dict[f"{set_name}/cm_{team_name}"] = wandb.plot.confusion_matrix(
                y_true=y_true_list, preds=y_pred_list, class_names=["No Goal", "Goal"]
            )
        except Exception as e:
            print(f"  WARNING: Could not plot confusion matrix for {team_name} on {set_name} set. Error: {e}")

        try:
            log_dict[f"{set_name}/pr_curve_{team_name}"] = wandb.plot.pr_curve(
                y_true_list, y_probas_for_plots, labels=["No Goal", "Goal"]
            )
            log_dict[f"{set_name}/roc_curve_{team_name}"] = wandb.plot.roc_curve(
                y_true_list, y_probas_for_plots, labels=["No Goal", "Goal"]
            )
        except Exception as e:
            print(f"  WARNING: Could not plot PR/ROC curves for {team_name} on {set_name} set. Error: {e}")

        if set_name == "test":
            wandb.summary.update({
                f"test_f1_{team_name}": f1,
                f"test_precision_{team_name}": precision,
                f"test_recall_{team_name}": recall
            })

        wandb.log(log_dict)

def log_feature_importance_plot(importance_df, title, wandb_key, top_n=30, use_all=False):
    """Creates a bar plot of the top 30 feature importances and logs it to W&B."""
    if not wandb.run:
        return
    
    # Select the top requested features for a clean plot
    top_df = importance_df
    if not use_all:
        top_df = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 12))
    sns.barplot(x='importance', y='feature', data=top_df, palette='viridis')
    plt.title(title, fontsize=16)
    plt.xlabel('Importance Score (Gain)', fontsize=12)
    plt.ylabel('Feature Name', fontsize=12)
    plt.tight_layout()
    
    # Log the plot as an image to Weights & Biases
    wandb.log({wandb_key: wandb.Image(plt)})
    
    # Close the plot to free up memory
    plt.close()

def find_optimal_threshold(y_true, y_pred_proba):
    """
    Finds the optimal decision threshold from a validation set to maximize the F1-score.
    """
    # Ensure inputs are NumPy arrays
    y_true_np = np.array(y_true)
    y_pred_proba_np = np.array(y_pred_proba)

    precisions, recalls, thresholds = precision_recall_curve(y_true_np, y_pred_proba_np)
    
    # Calculate F1 score for each threshold. Add a small epsilon to avoid division by zero.
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    
    # The last precision and recall values are 1. and 0. respectively and do not have a corresponding threshold.
    # We find the index of the maximum F1 score, excluding this last value.
    best_f1_idx = np.argmax(f1_scores[:-1])
    
    # The thresholds array is one element shorter, so this index is valid.
    best_threshold = thresholds[best_f1_idx]
    
    return best_threshold, f1_scores[best_f1_idx]    

# ====================== MAIN ======================
def main():
    args = parse_args()
    try:
        wandb.init(project=args.wandb_project, name=args.run_name, config=args)
    except Exception as e:
        print(f"--- Could not initialize W&B: {e}. Running without logging. ---")
        wandb.run = None

    train_dir, val_dir, test_dir = [os.path.join(args.data_dir, split) for split in ['train','val','test']]
    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.csv')]
    val_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.csv')]
    test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.csv')]

    X_train, y_orange_train, y_blue_train = RobustLazyLoader(train_files).to_dataframe()
    X_val, y_orange_val, y_blue_val = RobustLazyLoader(val_files).to_dataframe()
    X_test, y_orange_test, y_blue_test = RobustLazyLoader(test_files).to_dataframe()

    if X_train is None:
        print("CRITICAL ERROR: No training data could be loaded. Exiting.")
        return

    scale_pos_weight_orange = (y_orange_train == 0).sum() / (y_orange_train == 1).sum()
    scale_pos_weight_blue = (y_blue_train == 0).sum() / (y_blue_train == 1).sum()
    print(f"\nCalculated scale_pos_weight for Orange: {scale_pos_weight_orange:.2f}")
    print(f"Calculated scale_pos_weight for Blue: {scale_pos_weight_blue:.2f}")

    clf_orange = XGBClassifier(
        max_depth=args.max_depth, n_estimators=args.n_estimators, learning_rate=args.learning_rate,
        subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', tree_method='hist',
        n_jobs=args.num_threads, use_label_encoder=False, 
        scale_pos_weight=scale_pos_weight_orange,
        early_stopping_rounds=args.early_stopping_rounds
    )
    clf_blue = XGBClassifier(
        max_depth=args.max_depth, n_estimators=args.n_estimators, learning_rate=args.learning_rate,
        subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', tree_method='hist',
        n_jobs=args.num_threads, use_label_encoder=False, 
        scale_pos_weight=scale_pos_weight_blue,
        early_stopping_rounds=args.early_stopping_rounds
    )

    print("\n--- Training Orange Classifier ---")
    clf_orange.fit(X_train, y_orange_train, eval_set=[(X_val, y_orange_val)], verbose=True)
    
    print("\n--- Training Blue Classifier ---")
    clf_blue.fit(X_train, y_blue_train, eval_set=[(X_val, y_blue_val)], verbose=True)

    print("\n--- Evaluating on VALIDATION Set ---")
    evaluate_and_log(clf_orange, X_val, y_orange_val, "orange", set_name="val")
    evaluate_and_log(clf_blue, X_val, y_blue_val, "blue", set_name="val")

    print("\n--- Evaluating on TEST Set ---")
    evaluate_and_log(clf_orange, X_test, y_orange_test, "orange", set_name="test")
    evaluate_and_log(clf_blue, X_test, y_blue_test, "blue", set_name="test")

    print("\n--- Performing Threshold Optimization on VALIDATION Set ---")
    # Get the raw probabilities from the validation set (this is our "practice lap")
    val_probs_orange = clf_orange.predict_proba(X_val)[:, 1]
    val_probs_blue = clf_blue.predict_proba(X_val)[:, 1]

    # Find the optimal threshold for each model
    optimal_threshold_orange, max_f1_val_orange = find_optimal_threshold(y_orange_val, val_probs_orange)
    optimal_threshold_blue, max_f1_val_blue = find_optimal_threshold(y_blue_val, val_probs_blue)

    print(f"  Optimal Threshold for Orange: {optimal_threshold_orange:.4f} (achieved F1: {max_f1_val_orange:.4f} on val set)")
    print(f"  Optimal Threshold for Blue:   {optimal_threshold_blue:.4f} (achieved F1: {max_f1_val_blue:.4f} on val set)")
    
    print("\n--- Evaluating on TEST Set with OPTIMIZED Thresholds ---")

    # Get the raw probabilities from the test set
    test_probs_orange = clf_orange.predict_proba(X_test)[:, 1]
    test_probs_blue = clf_blue.predict_proba(X_test)[:, 1]

    # Apply the new, learned thresholds to the test set predictions
    test_preds_optimized_orange = (test_probs_orange > optimal_threshold_orange).astype(int)
    test_preds_optimized_blue = (test_probs_blue > optimal_threshold_blue).astype(int)

    # Calculate the final, optimized F1 scores
    final_f1_orange = f1_score(y_orange_test, test_preds_optimized_orange, zero_division=0)
    final_f1_blue = f1_score(y_blue_test, test_preds_optimized_blue, zero_division=0)

    print(f"  Final OPTIMIZED F1 Score (Orange): {final_f1_orange:.4f}")
    print(f"  Final OPTIMIZED F1 Score (Blue):   {final_f1_blue:.4f}")
    
    # Log these final, most important scores to the W&B summary
    if wandb.run:
        wandb.summary["optimal_threshold_orange"] = optimal_threshold_orange
        wandb.summary["optimal_threshold_blue"] = optimal_threshold_blue
        wandb.summary["optimized_test_f1_orange"] = final_f1_orange
        wandb.summary["optimized_test_f1_blue"] = final_f1_blue
    
    if wandb.run:
        print("\n--- Logging Feature Importance to W&B ---")
        df_importance_orange = pd.DataFrame({'feature': X_train.columns, 'importance': clf_orange.feature_importances_}).sort_values('importance', ascending=False)
        df_importance_blue = pd.DataFrame({'feature': X_train.columns, 'importance': clf_blue.feature_importances_}).sort_values('importance', ascending=False)
        wandb.log({
            "feature_importance_orange": wandb.Table(dataframe=df_importance_orange),
            "feature_importance_blue": wandb.Table(dataframe=df_importance_blue)
        })
        log_feature_importance_plot(df_importance_orange, "Top 30 Feature Importances (Orange Model)", "feature_importance_plot_orange", use_all=True)
        log_feature_importance_plot(df_importance_blue, "Top 30 Feature Importances (Blue Model)", "feature_importance_plot_blue", use_all=True)

        wandb.finish()

    if args.model_save_path:
        # Create the directory if it doesn't exist
        output_dir = os.path.dirname(args.model_save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Define paths for the two separate models
        orange_path = args.model_save_path.replace('.json', '_orange.json')
        blue_path = args.model_save_path.replace('.json', '_blue.json')

        clf_orange.save_model(orange_path)
        clf_blue.save_model(blue_path)
        print(f"\nModels saved successfully to:\n  - {orange_path}\n  - {blue_path}")

if __name__ == "__main__":
    main()