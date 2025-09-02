import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import linecache
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from xgboost import XGBClassifier
import wandb

# ====================== CONFIGURATION & CONSTANTS ======================
NUM_PLAYERS = 6; PLAYER_FEATURES = 13; GLOBAL_FEATURES = 14
TOTAL_FLAT_FEATURES = (NUM_PLAYERS * PLAYER_FEATURES) + GLOBAL_FEATURES
POS_MIN_X, POS_MAX_X = -4096, 4096; POS_MIN_Y, POS_MAX_Y = -6000, 6000; POS_MIN_Z, POS_MAX_Z = 0, 2044
VEL_MIN, VEL_MAX = -2300, 2300; BOOST_MIN, BOOST_MAX = 0, 100; BALL_VEL_MIN, BALL_VEL_MAX = -6000, 6000
BOOST_PAD_MIN, BOOST_PAD_MAX = 0, 10; DIST_MIN, DIST_MAX = 0, (8192**2 + 10240**2 + 2044**2)**0.5

def normalize(val, min_val, max_val):
    return (val - min_val) / (max_val - min_val + 1e-8)

# ====================== ARGUMENT PARSER ======================
def parse_args():
    parser = argparse.ArgumentParser(description="Rocket League Dual XGBoost Training (Compatible)")
    parser.add_argument("--data-dir", type=str, default=r"E:\\Raw RL Esports Replays\\Day 3 Swiss Stage\\Round 1\\split_dataset", help="Path to dataset root directory")
    parser.add_argument('--wandb-project', type=str, default="rl-goal-prediction-baselines", help="W&B project name.")
    parser.add_argument('--run-name', type=str, default=None, help="Custom name for the W&B run.")
    parser.add_argument('--max-depth', type=int, default=6, help='XGBoost max depth.')
    parser.add_argument('--n-estimators', type=int, default=200, help='Number of trees.')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='XGBoost learning rate.')
    parser.add_argument('--save-models', action='store_true', help='Save trained models as JSON.')
    parser.add_argument("--num-threads", type=int, default=4, help="Number of threads XGBoost should use")
    return parser.parse_args()

# ====================== DATA LOADER (Your working loader) ======================
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
            except Exception as e: print(f"\nWarning: Could not process file {path}. Skipping. Error: {e}")
        self.length = total_rows

    def to_numpy(self):
        X, y_orange, y_blue = [], [], []
        desc = f"Converting {os.path.basename(os.path.dirname(self.csv_paths[0]))} dataset to numpy"
        for idx in tqdm(range(len(self)), desc=desc):
            item = self.__getitem__(idx)
            if item is None: continue
            feat, oy, by = item
            X.append(feat); y_orange.append(oy); y_blue.append(by)
        return np.array(X), np.array(y_orange), np.array(y_blue)

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
            player_features = [item for i in range(NUM_PLAYERS) for item in [normalize(float(row[f'p{i}_pos_x']), POS_MIN_X, POS_MAX_X), normalize(float(row[f'p{i}_pos_y']), POS_MIN_Y, POS_MAX_Y), normalize(float(row[f'p{i}_pos_z']), POS_MIN_Z, POS_MAX_Z), normalize(float(row[f'p{i}_vel_x']), VEL_MIN, VEL_MAX), normalize(float(row[f'p{i}_vel_y']), VEL_MIN, VEL_MAX), normalize(float(row[f'p{i}_vel_z']), VEL_MIN, VEL_MAX), float(row[f'p{i}_forward_x']), float(row[f'p{i}_forward_y']), float(row[f'p{i}_forward_z']), normalize(float(row[f'p{i}_boost_amount']), BOOST_MIN, BOOST_MAX), float(row[f'p{i}_team']), float(row[f'p{i}_alive']), normalize(float(row[f'p{i}_dist_to_ball']), DIST_MIN, DIST_MAX)]]
            global_features = [normalize(float(row['ball_pos_x']), POS_MIN_X, POS_MAX_X), normalize(float(row['ball_pos_y']), POS_MIN_Y, POS_MAX_Y), normalize(float(row['ball_pos_z']), POS_MIN_Z, POS_MAX_Z), normalize(float(row['ball_vel_x']), BALL_VEL_MIN, BALL_VEL_MAX), normalize(float(row['ball_vel_y']), BALL_VEL_MIN, BALL_VEL_MAX), normalize(float(row['ball_vel_z']), BALL_VEL_MIN, BALL_VEL_MAX), normalize(float(row['boost_pad_0_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX), normalize(float(row['boost_pad_1_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX), normalize(float(row['boost_pad_2_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX), normalize(float(row['boost_pad_3_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX), normalize(float(row['boost_pad_4_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX), normalize(float(row['boost_pad_5_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX), float(row['ball_hit_team_num']), normalize(min(float(row['seconds_remaining']), 300.0), 0, 300)]
            features = np.array(player_features + global_features, dtype=np.float32)
            orange_y = int(float(row['team_1_goal_in_event_window']))
            blue_y = int(float(row['team_0_goal_in_event_window']))
            return features, orange_y, blue_y
        except: return None

# ====================== MAIN ======================
def main():
    args = parse_args()
    try:
        wandb.init(project=args.wandb_project, name=args.run_name, config=args)
    except Exception as e:
        print(f"--- Could not initialize W&B: {e}. Running without logging. ---"); wandb.run = None

    # Load datasets
    train_dir, val_dir, test_dir = [os.path.join(args.data_dir, split) for split in ['train','val','test']]
    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.csv')]
    val_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.csv')]
    test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.csv')]

    X_train, y_orange_train, y_blue_train = RobustLazyLoader(train_files).to_numpy()
    X_val, y_orange_val, y_blue_val = RobustLazyLoader(val_files).to_numpy()
    X_test, y_orange_test, y_blue_test = RobustLazyLoader(test_files).to_numpy()

    # --- FIX #1: Calculate scale_pos_weight for class imbalance ---
    scale_pos_weight_orange = (y_orange_train == 0).sum() / (y_orange_train == 1).sum()
    scale_pos_weight_blue = (y_blue_train == 0).sum() / (y_blue_train == 1).sum()
    print(f"\nCalculated scale_pos_weight for Orange: {scale_pos_weight_orange:.2f}")
    print(f"Calculated scale_pos_weight for Blue: {scale_pos_weight_blue:.2f}")

    # Train classifiers
    clf_orange = XGBClassifier(
        max_depth=args.max_depth, n_estimators=args.n_estimators, learning_rate=args.learning_rate,
        subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', tree_method='hist',
        n_jobs=args.num_threads, use_label_encoder=False, 
        scale_pos_weight=scale_pos_weight_orange # Add the imbalance parameter
    )
    clf_blue = XGBClassifier(
        max_depth=args.max_depth, n_estimators=args.n_estimators, learning_rate=args.learning_rate,
        subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', tree_method='hist',
        n_jobs=args.num_threads, use_label_encoder=False, 
        scale_pos_weight=scale_pos_weight_blue # Add the imbalance parameter
    )

    print("\n--- Training Orange Classifier ---")
    # --- FIX #2: REMOVED early_stopping_rounds ---
    clf_orange.fit(X_train, y_orange_train, eval_set=[(X_val, y_orange_val)], verbose=True)
    
    print("\n--- Training Blue Classifier ---")
    # --- FIX #2: REMOVED early_stopping_rounds ---
    clf_blue.fit(X_train, y_blue_train, eval_set=[(X_val, y_blue_val)], verbose=True)

    # Validation evaluation (your working logic)
    # ... (the rest of your script is perfect, so I will include it without changes)
    val_probs_o, val_probs_b = clf_orange.predict_proba(X_val)[:,1], clf_blue.predict_proba(X_val)[:,1]
    val_preds_o, val_preds_b = (val_probs_o > 0.5).astype(int), (val_probs_b > 0.5).astype(int)
    metrics = {"val": {"Orange": {"f1": f1_score(y_orange_val, val_preds_o), "precision": precision_score(y_orange_val, val_preds_o), "recall": recall_score(y_orange_val, val_preds_o)}, "Blue": {"f1": f1_score(y_blue_val, val_preds_b), "precision": precision_score(y_blue_val, val_preds_b), "recall": recall_score(y_blue_val, val_preds_b)}}}
    print("\nValidation Results:")
    for team in ["Orange", "Blue"]:
        m = metrics["val"][team]
        print(f"  {team} - F1: {m['f1']:.4f}, Precision: {m['precision']:.4f}, Recall: {m['recall']:.4f}")
    if wandb.run:
        class_names = ["No Goal", "Goal"]
        wandb.log({"val/f1_orange": metrics["val"]["Orange"]["f1"], "val/precision_orange": metrics["val"]["Orange"]["precision"], "val/recall_orange": metrics["val"]["Orange"]["recall"], "val/f1_blue": metrics["val"]["Blue"]["f1"], "val/precision_blue": metrics["val"]["Blue"]["precision"], "val/recall_blue": metrics["val"]["Blue"]["recall"]})
        wandb.log({"val/cm_orange": wandb.plot.confusion_matrix(y_true=y_orange_val, preds=val_preds_o, class_names=class_names), "val/cm_blue": wandb.plot.confusion_matrix(y_true=y_blue_val, preds=val_preds_b, class_names=class_names), "val/pr_curve_orange": wandb.plot.pr_curve(y_true=y_orange_val, y_probas=np.stack([1-val_probs_o, val_probs_o], axis=1), labels=class_names), "val/pr_curve_blue": wandb.plot.pr_curve(y_true=y_blue_val, y_probas=np.stack([1-val_probs_b, val_probs_b], axis=1), labels=class_names), "val/roc_curve_orange": wandb.plot.roc_curve(y_true=y_orange_val, y_probas=np.stack([1-val_probs_o, val_probs_o], axis=1), labels=class_names), "val/roc_curve_blue": wandb.plot.roc_curve(y_true=y_blue_val, y_probas=np.stack([1-val_probs_b, val_probs_b], axis=1), labels=class_names)})

    # Test evaluation
    test_probs_o, test_probs_b = clf_orange.predict_proba(X_test)[:,1], clf_blue.predict_proba(X_test)[:,1]
    test_preds_o, test_preds_b = (test_probs_o > 0.5).astype(int), (test_probs_b > 0.5).astype(int)
    metrics["test"] = {"Orange": {"f1": f1_score(y_orange_test, test_preds_o), "precision": precision_score(y_orange_test, test_preds_o), "recall": recall_score(y_orange_test, test_preds_o)}, "Blue": {"f1": f1_score(y_blue_test, test_preds_b), "precision": precision_score(y_blue_test, test_preds_b), "recall": recall_score(y_blue_test, test_preds_b)}}
    print("\n--- FINAL TEST RESULTS ---")
    for team in ["Orange", "Blue"]:
        m = metrics["test"][team]
        print(f"  {team} - F1: {m['f1']:.4f}, Precision: {m['precision']:.4f}, Recall: {m['recall']:.4f}")
    if wandb.run:
        wandb.summary.update({"test_f1_orange": metrics["test"]["Orange"]["f1"], "test_precision_orange": metrics["test"]["Orange"]["precision"], "test_recall_orange": metrics["test"]["Orange"]["recall"], "test_f1_blue": metrics["test"]["Blue"]["f1"], "test_precision_blue": metrics["test"]["Blue"]["precision"], "test_recall_blue": metrics["test"]["Blue"]["recall"]})
        wandb.log({"test/cm_orange": wandb.plot.confusion_matrix(y_true=y_orange_test, preds=test_preds_o, class_names=["No Goal","Goal"]), "test/cm_blue": wandb.plot.confusion_matrix(y_true=y_blue_test, preds=test_preds_b, class_names=["No Goal","Goal"])})
        
        # Add feature importance logging
        feature_names = X_train.columns if isinstance(X_train, pd.DataFrame) else [f'f{i}' for i in range(X_train.shape[1])]
        df_importance_orange = pd.DataFrame({'feature': feature_names, 'importance': clf_orange.feature_importances_}).sort_values('importance', ascending=False).head(30)
        df_importance_blue = pd.DataFrame({'feature': feature_names, 'importance': clf_blue.feature_importances_}).sort_values('importance', ascending=False).head(30)
        wandb.log({"feature_importance_orange": wandb.Table(dataframe=df_importance_orange), "feature_importance_blue": wandb.Table(dataframe=df_importance_blue)})

    if wandb.run:
        wandb.finish()

    if args.save_models:
        os.makedirs("./models", exist_ok=True)
        clf_orange.save_model("./models/orange_model.json")
        clf_blue.save_model("./models/blue_model.json")
        print("\nModels saved to ./models/ directory")

if __name__ == "__main__":
    main()