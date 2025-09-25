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

# ====================== CONFIGURATION & MODEL DEFINITIONS ======================
NUM_PLAYERS = 6; PLAYER_FEATURES = 13; GLOBAL_FEATURES = 14 
TOTAL_FLAT_FEATURES = (NUM_PLAYERS * PLAYER_FEATURES) + GLOBAL_FEATURES

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=TOTAL_FLAT_FEATURES):
        super(BaselineMLP, self).__init__(); self.body = nn.Sequential(nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5)); self.orange_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid()); self.blue_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
    def forward(self, x): x_p = self.body(x); return self.orange_head(x_p), self.blue_head(x_p)

# ====================== HELPER FUNCTIONS ======================
def normalize_dataframe_corrected(raw_df):
    """Replicates the EXACT same preprocessing as your MLP training."""
    POS_MIN_X, POS_MAX_X = -4096, 4096; POS_MIN_Y, POS_MAX_Y = -6000, 6000; POS_MIN_Z, POS_MAX_Z = 0, 2044
    VEL_MIN, VEL_MAX = -2300, 2300; BOOST_MIN, BOOST_MAX = 0, 100; BALL_VEL_MIN, BALL_VEL_MAX = -6000, 6000
    BOOST_PAD_MIN, BOOST_PAD_MAX = 0, 10; DIST_MIN, DIST_MAX = 0, (8192**2 + 10240**2 + 2044**2)**0.5
    
    def normalize(val, min_val, max_val): 
        return (val - min_val) / (max_val - min_val + 1e-8)
    
    all_features = []
    for _, row in raw_df.iterrows():
        try:
            # Replicate the EXACT same feature construction as training
            player_features = []
            for i in range(NUM_PLAYERS):
                player_features.extend([
                    normalize(float(row[f'p{i}_pos_x']), POS_MIN_X, POS_MAX_X),
                    normalize(float(row[f'p{i}_pos_y']), POS_MIN_Y, POS_MAX_Y),
                    normalize(float(row[f'p{i}_pos_z']), POS_MIN_Z, POS_MAX_Z),
                    normalize(float(row[f'p{i}_vel_x']), VEL_MIN, VEL_MAX),
                    normalize(float(row[f'p{i}_vel_y']), VEL_MIN, VEL_MAX),
                    normalize(float(row[f'p{i}_vel_z']), VEL_MIN, VEL_MAX),
                    float(row[f'p{i}_forward_x']),      # Raw (as in training)
                    float(row[f'p{i}_forward_y']),      # Raw
                    float(row[f'p{i}_forward_z']),      # Raw
                    normalize(float(row[f'p{i}_boost_amount']), BOOST_MIN, BOOST_MAX),
                    float(row[f'p{i}_team']),           # Raw
                    float(row[f'p{i}_alive']),          # Raw
                    normalize(float(row[f'p{i}_dist_to_ball']), DIST_MIN, DIST_MAX)
                ])
            
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
                float(row['ball_hit_team_num']),        # Raw
                normalize(min(float(row['seconds_remaining']), 300.0), 0, 300)
            ]
            
            all_features.append(player_features + global_features)
        except (ValueError, KeyError):
            continue
    
    return np.array(all_features)

def get_xgb_feature_columns(df_chunk):
    """Get the EXACT same features that XGBoost was trained on."""
    # These are ALL the features your XGBoost models expect (based on your training script)
    feature_cols = [col for col in df_chunk.columns if any([
        col.startswith('p'),           # Player features (pos, vel, forward, boost, team, alive, dist)
        col.startswith('ball'),        # Ball features (pos, vel)
        col.startswith('boost_pad'),   # Boost pad respawn times
        col == 'seconds_remaining'     # Time remaining
    ])]
    return feature_cols

def load_all_models(args, device):
    """Loads all three trained models."""
    try:
        model_mlp = BaselineMLP(input_dim=TOTAL_FLAT_FEATURES).to(device); print(f"  Loading MLP model from: {args.model_path_mlp}"); checkpoint = torch.load(args.model_path_mlp, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint: model_mlp.load_state_dict(checkpoint['model_state'])
        else: model_mlp.load_state_dict(checkpoint)
        model_mlp.eval(); print("  MLP model loaded successfully.")
        model_xgb_orange = xgb.XGBClassifier(); print(f"  Loading XGBoost Orange model from: {args.model_path_xgb_orange}"); model_xgb_orange.load_model(args.model_path_xgb_orange); print("  XGBoost Orange model loaded successfully.")
        model_xgb_blue = xgb.XGBClassifier(); print(f"  Loading XGBoost Blue model from: {args.model_path_xgb_blue}"); model_xgb_blue.load_model(args.model_path_xgb_blue); print("  XGBoost Blue model loaded successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR loading models: {e}"); return None, None, None
    return model_mlp, model_xgb_orange, model_xgb_blue

def calculate_time_to_goal(df):
    """Processes a complete DataFrame and calculates the time-to-goal for each positive sample."""
    print("\n--- Calculating time-to-goal for all positive samples... ---")
    df['time_to_goal_orange'] = np.nan; df['time_to_goal_blue'] = np.nan
    for _, group in tqdm(df.groupby('replay_id'), desc="Processing replays"):
        last_goal_time_o = -1; last_score_diff = None
        for i in range(len(group) - 1, -1, -1):
            idx = group.index[i]; row = group.loc[idx]
            if row['team_1_goal_in_event_window'] == 1:
                is_last_row = (i == len(group) - 1); next_row_is_zero = not is_last_row and group.loc[group.index[i+1], 'team_1_goal_in_event_window'] == 0
                score_changed = last_score_diff is not None and row['score_difference'] != last_score_diff
                if last_goal_time_o == -1 or next_row_is_zero or score_changed: last_goal_time_o = row['time']
                df.loc[idx, 'time_to_goal_orange'] = last_goal_time_o - row['time']
            else: last_goal_time_o = -1
            last_score_diff = row['score_difference']
        last_goal_time_b = -1; last_score_diff = None
        for i in range(len(group) - 1, -1, -1):
            idx = group.index[i]; row = group.loc[idx]
            if row['team_0_goal_in_event_window'] == 1:
                is_last_row = (i == len(group) - 1); next_row_is_zero = not is_last_row and group.loc[group.index[i+1], 'team_0_goal_in_event_window'] == 0
                score_changed = last_score_diff is not None and row['score_difference'] != last_score_diff
                if last_goal_time_b == -1 or next_row_is_zero or score_changed: last_goal_time_b = row['time']
                df.loc[idx, 'time_to_goal_blue'] = last_goal_time_b - row['time']
            else: last_goal_time_b = -1
            last_score_diff = row['score_difference']
    return df

# ====================== MAIN ANALYSIS ======================
def main():
    parser = argparse.ArgumentParser(description="Analyze Ensemble Predictions by Time-to-Goal on the TEST set")
    parser.add_argument('--data-dir', type=str, required=True, help='Path to the directory containing the dataset splits.')
    parser.add_argument('--model-path-mlp', type=str, required=True, help='Path to the saved MLP model (.pth file).')
    parser.add_argument('--model-path-xgb-orange', type=str, required=True, help='Path to the saved ORANGE XGBoost model (.json file).')
    parser.add_argument('--model-path-xgb-blue', type=str, required=True, help='Path to the saved BLUE XGBoost model (.json file).')
    parser.add_argument('--output-csv', type=str, default='./time_to_goal_analysis.csv', help='Path to save the final analysis CSV.')
    parser.add_argument('--output-plot', type=str, default='./time_to_goal_analysis.png', help='Path to save the final analysis plot.')
    parser.add_argument('--batch-size', type=int, default=2048, help='Batch size for MLP evaluation.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 1. Load Models ---
    model_mlp, model_xgb_orange, model_xgb_blue = load_all_models(args, device)
    if model_mlp is None: return

    # --- 2. Get Predictions on TEST Set (Memory-Safe) ---
    print("\n--- Getting Predictions from All Models on TEST SET (Memory-Efficiently) ---")
    
    data_dir = os.path.join(args.data_dir, 'test')
    data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    results_dfs = []
    for file_path in tqdm(data_files, desc="Processing Test Chunks"):
        df_chunk = pd.read_csv(file_path).dropna()
        if df_chunk.empty: continue
        
        # FIXED: Use the same feature selection for both models
        feature_cols = get_xgb_feature_columns(df_chunk)
        X_raw = df_chunk[feature_cols]
        X_norm_np = normalize_dataframe_corrected(df_chunk)  # Fixed  # MLP gets normalized data

        with torch.no_grad():
            features_tensor = torch.tensor(X_norm_np, dtype=torch.float32).to(device)
            mlp_orange_prob, mlp_blue_prob = model_mlp(features_tensor)
        
        # XGBoost gets RAW data (same as during training)
        xgb_orange_prob = model_xgb_orange.predict_proba(X_raw)[:, 1]
        xgb_blue_prob = model_xgb_blue.predict_proba(X_raw)[:, 1]

        chunk_results = df_chunk[['replay_id', 'time', 'score_difference', 'team_0_goal_in_event_window', 'team_1_goal_in_event_window']].copy()
        chunk_results['ensemble_prob_orange'] = (mlp_orange_prob.cpu().numpy().flatten() + xgb_orange_prob) / 2.0
        chunk_results['ensemble_prob_blue'] = (mlp_blue_prob.cpu().numpy().flatten() + xgb_blue_prob) / 2.0
        results_dfs.append(chunk_results)

    # --- 3. Combine Results and Calculate Time-to-Goal ---
    final_results_df = pd.concat(results_dfs, ignore_index=True)
    final_results_df = calculate_time_to_goal(final_results_df)
    
    # --- 4. Analyze Orange Team ---
    orange_positives = final_results_df[final_results_df['time_to_goal_orange'].notna()].copy()
    bins = np.arange(0, 5.2, 0.2); labels = [f"{i:.1f}-{i+0.2:.1f}" for i in bins[:-1]]
    orange_positives['time_bin'] = pd.cut(orange_positives['time_to_goal_orange'], bins=bins, labels=labels, right=False)
    def success_rate(x): return (x > 0.5).mean()
    orange_stats = orange_positives.groupby('time_bin')['ensemble_prob_orange'].agg(['mean', 'std', 'min', 'max', 'count', success_rate]).reset_index()
    orange_stats.columns = ['time_bin', 'pred_mean', 'pred_std', 'pred_min', 'pred_max', 'sample_count', 'success_rate']

    # --- 5. Analyze Blue Team & Save ---
    blue_positives = final_results_df[final_results_df['time_to_goal_blue'].notna()].copy()
    blue_positives['time_bin'] = pd.cut(blue_positives['time_to_goal_blue'], bins=bins, labels=labels, right=False)
    blue_stats = blue_positives.groupby('time_bin')['ensemble_prob_blue'].agg(['mean', 'std', 'min', 'max', 'count', success_rate]).reset_index()
    blue_stats.columns = ['time_bin', 'pred_mean', 'pred_std', 'pred_min', 'pred_max', 'sample_count', 'success_rate']
    
    final_agg_df = pd.concat([orange_stats, blue_stats], keys=['Orange', 'Blue'], names=['team']).reset_index()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    final_agg_df.to_csv(args.output_csv, index=False)
    print(f"\nAnalysis complete. Results saved to {args.output_csv}")
    
    # --- 6. Plot Results ---
    print("--- Generating analysis plot ---")
    fig, axes = plt.subplots(2, 1, figsize=(15, 12), sharex=True); sns.set_theme(style="whitegrid")
    
    # Top Plot: Confidence
    sns.lineplot(ax=axes[0], data=orange_stats, x='time_bin', y='pred_mean', color='orange', label='Orange Team Avg. Prediction'); axes[0].fill_between(orange_stats.index, orange_stats['pred_min'], orange_stats['pred_max'], color='orange', alpha=0.2)
    sns.lineplot(ax=axes[0], data=blue_stats, x='time_bin', y='pred_mean', color='blue', label='Blue Team Avg. Prediction'); axes[0].fill_between(blue_stats.index, blue_stats['pred_min'], blue_stats['pred_max'], color='blue', alpha=0.2)
    axes[0].set_ylabel('Avg. Predicted Probability'); axes[0].set_title('Model Prediction Confidence vs. Time-to-Goal (Test Set)', fontsize=16); axes[0].legend()
    
    # Bottom Plot: Success Rate
    sns.lineplot(ax=axes[1], data=orange_stats, x='time_bin', y='success_rate', color='orange', label='Orange Team Success Rate'); sns.lineplot(ax=axes[1], data=blue_stats, x='time_bin', y='success_rate', color='blue', label='Blue Team Success Rate')
    ax2 = axes[1].twinx(); sns.barplot(ax=ax2, data=orange_stats, x='time_bin', y='sample_count', color='orange', alpha=0.2); sns.barplot(ax=ax2, data=blue_stats, x='time_bin', y='sample_count', color='blue', alpha=0.2)
    ax2.set_ylabel('Number of Samples'); axes[1].set_ylabel('Success Rate (Precision at Th=0.5)'); axes[1].set_xlabel('Time Before Goal (Seconds)'); axes[1].set_title('Model Success Rate vs. Time-to-Goal (Test Set)', fontsize=16)
    
    # Improve x-axis tick readability
    tick_positions = np.arange(0, len(labels), 5); tick_labels = [labels[i] for i in tick_positions]
    axes[1].set_xticks(tick_positions); axes[1].set_xticklabels(tick_labels, rotation=45, ha="right")
    
    plt.tight_layout(); plt.savefig(args.output_plot, dpi=150); print(f"Analysis plot saved to {args.output_plot}")

if __name__ == '__main__':
    main()