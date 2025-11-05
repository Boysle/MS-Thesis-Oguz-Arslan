import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import linecache
import time  # ##### NEW/FIXED #####

# ##### NEW/FIXED #####
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, confusion_matrix, accuracy_score, average_precision_score
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
    parser = argparse.ArgumentParser(description="Rocket League Dual XGBoost Training (Upgraded)")
    parser.add_argument("--data-dir", type=str, default=r"E:\\Raw RL Esports Replays\\Day 3 Swiss Stage\\Round 1\\split_dataset", help="Path to dataset root")
    parser.add_argument('--wandb-project', type=str, default="rl-xgboost", help="W&B project name.")
    parser.add_argument('--run-name', type=str, default=None, help="Custom name for the W&B run.")
    parser.add_argument('--max-depth', type=int, default=6, help='XGBoost max depth.')
    
    ##### NEW/FIXED #####
    parser.add_argument('--n-estimators-max', type=int, default=5000, help='Maximum number of trees.')
    parser.add_argument('--learning-rate', type=float, default=0.05, help='XGBoost learning rate.')
    parser.add_argument('--early-stopping-rounds', type=int, default=100, help='Early stopping patience.')
    
    parser.add_argument("--num-threads", type=int, default=-1, help="Number of threads. -1 uses all cores.")
    parser.add_argument('--model-save-path', type=str, default=None, help='Base path to save the final models (e.g., /path/to/xgb.json).')
    return parser.parse_args()

# ====================== DATA LOADER ======================
class XGBoostDataLoader:
    # ##### NEW/FIXED #####: Simplified this class. XGBoost needs all data at once.
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
        print(f"\n--- Indexing complete. Total samples: {self.length} ---")

    def to_dataframe(self):
        desc = f"Loading {os.path.basename(os.path.dirname(self.csv_paths[0]))} data into memory"
        print(f"--- {desc}... ---")
        df_list = [pd.read_csv(info['path']) for info in tqdm(self.file_info, desc=desc)]
        if not df_list: return None, None, None, None
        
        full_df = pd.concat(df_list, ignore_index=True)
        
        # Manually normalize features
        print("--- Normalizing features... ---")
        
        # Player features
        for i in range(NUM_PLAYERS):
            full_df[f'p{i}_pos_x'] = normalize(full_df[f'p{i}_pos_x'], POS_MIN_X, POS_MAX_X)
            full_df[f'p{i}_pos_y'] = normalize(full_df[f'p{i}_pos_y'], POS_MIN_Y, POS_MAX_Y)
            full_df[f'p{i}_pos_z'] = normalize(full_df[f'p{i}_pos_z'], POS_MIN_Z, POS_MAX_Z)
            full_df[f'p{i}_vel_x'] = normalize(full_df[f'p{i}_vel_x'], VEL_MIN, VEL_MAX)
            full_df[f'p{i}_vel_y'] = normalize(full_df[f'p{i}_vel_y'], VEL_MIN, VEL_MAX)
            full_df[f'p{i}_vel_z'] = normalize(full_df[f'p{i}_vel_z'], VEL_MIN, VEL_MAX)
            full_df[f'p{i}_boost_amount'] = normalize(full_df[f'p{i}_boost_amount'], BOOST_MIN, BOOST_MAX)
            full_df[f'p{i}_dist_to_ball'] = normalize(full_df[f'p{i}_dist_to_ball'], DIST_MIN, DIST_MAX)
        
        # Ball features
        full_df['ball_pos_x'] = normalize(full_df['ball_pos_x'], POS_MIN_X, POS_MAX_X)
        full_df['ball_pos_y'] = normalize(full_df['ball_pos_y'], POS_MIN_Y, POS_MAX_Y)
        full_df['ball_pos_z'] = normalize(full_df['ball_pos_z'], POS_MIN_Z, POS_MAX_Z)
        full_df['ball_vel_x'] = normalize(full_df['ball_vel_x'], BALL_VEL_MIN, BALL_VEL_MAX)
        full_df['ball_vel_y'] = normalize(full_df['ball_vel_y'], BALL_VEL_MIN, BALL_VEL_MAX)
        full_df['ball_vel_z'] = normalize(full_df['ball_vel_z'], BALL_VEL_MIN, BALL_VEL_MAX)

        # Global features
        for i in range(6): # 6 boost pads
            full_df[f'boost_pad_{i}_respawn'] = normalize(full_df[f'boost_pad_{i}_respawn'], BOOST_PAD_MIN, BOOST_PAD_MAX)
        full_df['seconds_remaining'] = normalize(np.minimum(full_df['seconds_remaining'], 300.0), 0, 300)
        
        # Drop any rows that became NaN during processing
        full_df = full_df.dropna()
        if full_df.empty: return None, None, None, None
        
        feature_cols = [col for col in self.header if col.startswith('p') or col.startswith('ball_') or col.startswith('boost_pad') or col == 'seconds_remaining']
        # Ensure correct order and presence
        feature_cols_present = [col for col in feature_cols if col in full_df.columns]
        
        X = full_df[feature_cols_present]
        y_orange = full_df['team_1_goal_in_event_window']
        y_blue = full_df['team_0_goal_in_event_window']
        
        print("--- Data loading and normalization complete. ---")
        return X, y_orange, y_blue, feature_cols_present

# ====================== EVALUATION ======================
# ##### NEW/FIXED #####: Removed old `evaluate_and_log` function.

def log_feature_importance_plot(importance_df, title, wandb_key, top_n=30):
    if not wandb.run: return
    top_df = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 10)) # Adjusted size
    sns.barplot(x='importance', y='feature', data=top_df, palette='viridis')
    plt.title(title, fontsize=16)
    plt.xlabel('Importance Score (Gain)', fontsize=12)
    plt.ylabel('Feature Name', fontsize=12)
    plt.tight_layout()
    wandb.log({wandb_key: wandb.Image(plt)})
    plt.close()

def find_optimal_threshold(y_true, y_pred_proba):
    y_true_np = np.array(y_true)
    y_pred_proba_np = np.array(y_pred_proba)
    precisions, recalls, thresholds = precision_recall_curve(y_true_np, y_pred_proba_np)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_f1_idx = np.argmax(f1_scores[:-1])
    return thresholds[best_f1_idx], f1_scores[best_f1_idx]      

# ====================== MAIN ======================
def main():
    ##### NEW/FIXED #####
    start_time = time.time()
    
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

    X_train, y_orange_train, y_blue_train, feature_names = XGBoostDataLoader(train_files).to_dataframe()
    X_val, y_orange_val, y_blue_val, _ = XGBoostDataLoader(val_files).to_dataframe()
    X_test, y_orange_test, y_blue_test, _ = XGBoostDataLoader(test_files).to_dataframe()

    if X_train is None:
        print("CRITICAL ERROR: No training data could be loaded. Exiting.")
        return

    scale_pos_weight_orange = (y_orange_train == 0).sum() / (y_orange_train == 1).sum()
    scale_pos_weight_blue = (y_blue_train == 0).sum() / (y_blue_train == 1).sum()
    print(f"\nCalculated scale_pos_weight for Orange: {scale_pos_weight_orange:.2f}")
    print(f"Calculated scale_pos_weight for Blue: {scale_pos_weight_blue:.2f}")

    clf_orange = XGBClassifier(
        max_depth=args.max_depth,
        n_estimators=args.n_estimators_max, # Use max
        learning_rate=args.learning_rate,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', tree_method='hist', # Use 'hist' for speed
        n_jobs=args.num_threads, use_label_encoder=False, 
        scale_pos_weight=scale_pos_weight_orange,
        early_stopping_rounds=args.early_stopping_rounds # Use patience
    )
    clf_blue = XGBClassifier(
        max_depth=args.max_depth,
        n_estimators=args.n_estimators_max, # Use max
        learning_rate=args.learning_rate,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', tree_method='hist',
        n_jobs=args.num_threads, use_label_encoder=False, 
        scale_pos_weight=scale_pos_weight_blue,
        early_stopping_rounds=args.early_stopping_rounds # Use patience
    )

    print("\n--- Training Orange Classifier ---")
    clf_orange.fit(X_train, y_orange_train, eval_set=[(X_val, y_orange_val)], verbose=50) # Print every 50 trees
    
    print("\n--- Training Blue Classifier ---")
    clf_blue.fit(X_train, y_blue_train, eval_set=[(X_val, y_blue_val)], verbose=50) # Print every 50 trees

    ##### NEW/FIXED #####
    print("\n--- Training Complete ---")
    print(f"  Orange Model Best Iteration: {clf_orange.best_iteration} (LogLoss: {clf_orange.best_score:.4f})")
    print(f"  Blue Model Best Iteration:   {clf_blue.best_iteration} (LogLoss: {clf_blue.best_score:.4f})")

    # --- Step 1: Find Optimal Threshold on the FULL Validation Set ---
    print("\n--- Determining optimal thresholds on the validation set... ---")
    val_probs_orange = clf_orange.predict_proba(X_val)[:, 1]
    val_probs_blue = clf_blue.predict_proba(X_val)[:, 1]
    
    optimal_threshold_orange, _ = find_optimal_threshold(y_orange_val, val_probs_orange)
    optimal_threshold_blue, _ = find_optimal_threshold(y_blue_val, val_probs_blue)

    print(f"  Optimal Threshold (Orange): {optimal_threshold_orange:.4f}")
    print(f"  Optimal Threshold (Blue):   {optimal_threshold_blue:.4f}")

    # --- Step 2: Run Evaluation on the Test Set ---
    print("\n--- Running final evaluation on the test set... ---")
    y_true_o, y_prob_o = y_orange_test, clf_orange.predict_proba(X_test)[:, 1]
    y_true_b, y_prob_b = y_blue_test, clf_blue.predict_proba(X_test)[:, 1]

    # --- Default @ 0.5 ---
    preds_def_o = (y_prob_o > 0.5).astype(int)
    preds_def_b = (y_prob_b > 0.5).astype(int)
    tn_def_o, fp_def_o, fn_def_o, tp_def_o = confusion_matrix(y_true_o, preds_def_o, labels=[0,1]).ravel()
    f1_def_o = f1_score(y_true_o, preds_def_o, zero_division=0)
    prec_def_o = precision_score(y_true_o, preds_def_o, zero_division=0)
    rec_def_o = recall_score(y_true_o, preds_def_o, zero_division=0)
    acc_def_o = accuracy_score(y_true_o, preds_def_o)
    tn_def_b, fp_def_b, fn_def_b, tp_def_b = confusion_matrix(y_true_b, preds_def_b, labels=[0,1]).ravel()
    f1_def_b = f1_score(y_true_b, preds_def_b, zero_division=0)
    prec_def_b = precision_score(y_true_b, preds_def_b, zero_division=0)
    rec_def_b = recall_score(y_true_b, preds_def_b, zero_division=0)
    acc_def_b = accuracy_score(y_true_b, preds_def_b)

    # --- Optimized ---
    preds_opt_o = (y_prob_o > optimal_threshold_orange).astype(int)
    preds_opt_b = (y_prob_b > optimal_threshold_blue).astype(int)
    tn_opt_o, fp_opt_o, fn_opt_o, tp_opt_o = confusion_matrix(y_true_o, preds_opt_o, labels=[0,1]).ravel()
    f1_opt_o = f1_score(y_true_o, preds_opt_o, zero_division=0)
    prec_opt_o = precision_score(y_true_o, preds_opt_o, zero_division=0)
    rec_opt_o = recall_score(y_true_o, preds_opt_o, zero_division=0)
    acc_opt_o = accuracy_score(y_true_o, preds_opt_o)
    tn_opt_b, fp_opt_b, fn_opt_b, tp_opt_b = confusion_matrix(y_true_b, preds_opt_b, labels=[0,1]).ravel()
    f1_opt_b = f1_score(y_true_b, preds_opt_b, zero_division=0)
    prec_opt_b = precision_score(y_true_b, preds_opt_b, zero_division=0)
    rec_opt_b = recall_score(y_true_b, preds_opt_b, zero_division=0)
    acc_opt_b = accuracy_score(y_true_b, preds_opt_b)

    # --- AUPRC (Threshold-Independent) ---
    auprc_o = average_precision_score(y_true_o, y_prob_o)
    auprc_b = average_precision_score(y_true_b, y_prob_b)

    # --- Step 3: New Print Block ---
    print("\n--- FINAL TEST RESULTS ---")
    print("\n-- Default @ 0.5 Threshold --")
    print(f"  Orange Team: F1: {f1_def_o:.4f} | P: {prec_def_o:.4f} | R: {rec_def_o:.4f} | Acc: {acc_def_o:.4f}")
    print(f"    -> TP: {tp_def_o} | TN: {tn_def_o} | FP: {fp_def_o} | FN: {fn_def_o}")
    print(f"  Blue Team:   F1: {f1_def_b:.4f} | P: {prec_def_b:.4f} | R: {rec_def_b:.4f} | Acc: {acc_def_b:.4f}")
    print(f"    -> TP: {tp_def_b} | TN: {tn_def_b} | FP: {fp_def_b} | FN: {fn_def_b}")
    print("\n-- Optimized Threshold --")
    print(f"  Orange Team (@ {optimal_threshold_orange:.3f}): F1: {f1_opt_o:.4f} | P: {prec_opt_o:.4f} | R: {rec_opt_o:.4f} | Acc: {acc_opt_o:.4f}")
    print(f"    -> TP: {tp_opt_o} | TN: {tn_opt_o} | FP: {fp_opt_o} | FN: {fn_opt_o}")
    print(f"  Blue Team   (@ {optimal_threshold_blue:.3f}): F1: {f1_opt_b:.4f} | P: {prec_opt_b:.4f} | R: {rec_opt_b:.4f} | Acc: {acc_opt_b:.4f}")
    print(f"    -> TP: {tp_opt_b} | TN: {tn_opt_b} | FP: {fp_opt_b} | FN: {fn_opt_b}")
    print("\n-- Threshold-Independent --")
    print(f"  AUPRC (Orange): {auprc_o:.4f}")
    print(f"  AUPRC (Blue):   {auprc_b:.4f}")

    # --- Step 4: Log Final Summary to W&B ---
    if wandb.run:
        print("\n--- Logging final summary to W&B ---")
        # Log training results
        wandb.summary["best_iteration_orange"] = clf_orange.best_iteration
        wandb.summary["best_val_logloss_orange"] = clf_orange.best_score
        wandb.summary["best_iteration_blue"] = clf_blue.best_iteration
        wandb.summary["best_val_logloss_blue"] = clf_blue.best_score
        
        # Log Default 0.5 scores
        wandb.summary["default_test_f1_orange"] = f1_def_o
        wandb.summary["default_test_precision_orange"] = prec_def_o
        wandb.summary["default_test_recall_orange"] = rec_def_o
        wandb.summary["default_test_accuracy_orange"] = acc_def_o
        wandb.summary["default_test_tp_orange"] = tp_def_o
        wandb.summary["default_test_tn_orange"] = tn_def_o
        wandb.summary["default_test_f1_blue"] = f1_def_b
        wandb.summary["default_test_precision_blue"] = prec_def_b
        wandb.summary["default_test_recall_blue"] = rec_def_b
        wandb.summary["default_test_accuracy_blue"] = acc_def_b
        wandb.summary["default_test_tp_blue"] = tp_def_b
        wandb.summary["default_test_tn_blue"] = tn_def_b

        # Log Optimized scores
        wandb.summary["optimal_threshold_orange"] = optimal_threshold_orange
        wandb.summary["optimized_test_f1_orange"] = f1_opt_o
        wandb.summary["optimized_test_precision_orange"] = prec_opt_o
        wandb.summary["optimized_test_recall_orange"] = rec_opt_o
        wandb.summary["optimized_test_accuracy_orange"] = acc_opt_o
        wandb.summary["optimized_test_auprc_orange"] = auprc_o
        wandb.summary["optimized_test_tp_orange"] = tp_opt_o
        wandb.summary["optimized_test_tn_orange"] = tn_opt_o

        wandb.summary["optimal_threshold_blue"] = optimal_threshold_blue
        wandb.summary["optimized_test_f1_blue"] = f1_opt_b
        wandb.summary["optimized_test_precision_blue"] = prec_opt_b
        wandb.summary["optimized_test_recall_blue"] = rec_opt_b
        wandb.summary["optimized_test_accuracy_blue"] = acc_opt_b
        wandb.summary["optimized_test_auprc_blue"] = auprc_b
        wandb.summary["optimized_test_tp_blue"] = tp_opt_b
        wandb.summary["optimized_test_tn_blue"] = tn_opt_b
        
        # Log feature importance
        print("\n--- Logging Feature Importance to W&B ---")
        df_importance_orange = pd.DataFrame({'feature': feature_names, 'importance': clf_orange.feature_importances_}).sort_values('importance', ascending=False)
        df_importance_blue = pd.DataFrame({'feature': feature_names, 'importance': clf_blue.feature_importances_}).sort_values('importance', ascending=False)
        wandb.log({
            "feature_importance_orange_table": wandb.Table(dataframe=df_importance_orange),
            "feature_importance_blue_table": wandb.Table(dataframe=df_importance_blue)
        })
        log_feature_importance_plot(df_importance_orange, "Top 30 Feature Importances (Orange Model)", "feature_importance_plot_orange")
        log_feature_importance_plot(df_importance_blue, "Top 30 Feature Importances (Blue Model)", "feature_importance_plot_blue")

    if args.model_save_path:
        output_dir = os.path.dirname(args.model_save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        # Use a base name to save two models
        base_name = args.model_save_path.replace('.json', '')
        orange_path = f"{base_name}_orange.json"
        blue_path = f"{base_name}_blue.json"
        
        clf_orange.save_model(orange_path)
        clf_blue.save_model(blue_path)
        print(f"\nModels saved successfully to:\n  - {orange_path}\n  - {blue_path}")

    end_time = time.time()
    total_seconds = end_time - start_time
    print(f"\n--- Total Run Time: {total_seconds // 3600:.0f}h {(total_seconds % 3600) // 60:.0f}m {total_seconds % 60:.2f}s ---")
    if wandb.run:
        wandb.summary["total_run_time_seconds"] = total_seconds
        wandb.finish()

if __name__ == "__main__":
    main()