import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import torch
import torch.nn as nn
import xgboost as xgb

# ====================== CONFIGURATION & MODEL DEFINITIONS (Copied from analysis scripts) ======================
NUM_PLAYERS = 6; PLAYER_FEATURES = 13; GLOBAL_FEATURES = 14 
TOTAL_FLAT_FEATURES = (NUM_PLAYERS * PLAYER_FEATURES) + GLOBAL_FEATURES

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=TOTAL_FLAT_FEATURES):
        super(BaselineMLP, self).__init__(); self.body = nn.Sequential(nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5)); self.orange_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid()); self.blue_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
    def forward(self, x): x_p = self.body(x); return self.orange_head(x_p), self.blue_head(x_p)

# These constants should be defined globally at the top of your script
POS_MIN_X, POS_MAX_X = -4096, 4096; POS_MIN_Y, POS_MAX_Y = -6000, 6000; POS_MIN_Z, POS_MAX_Z = 0, 2044
VEL_MIN, VEL_MAX = -2300, 2300; BOOST_MIN, BOOST_MAX = 0, 100; BALL_VEL_MIN, BALL_VEL_MAX = -6000, 6000
BOOST_PAD_MIN, BOOST_PAD_MAX = 0, 10; DIST_MIN, DIST_MAX = 0, (8192**2 + 10240**2 + 2044**2)**0.5

def normalize(val, min_val, max_val):
    return (val - min_val) / (max_val - min_val + 1e-8)

def normalize_dataframe_for_mlp(raw_df_chunk):
    """
    Takes a raw DataFrame chunk and returns a fully normalized NumPy array
    with the exact same logic as the training scripts.
    """
    # Work on a copy to avoid modifying the original DataFrame in memory
    X_norm = raw_df_chunk.copy()

    # Loop through each column and apply the correct normalization rule
    for col in X_norm.columns:
        if 'pos_x' in col or 'ball_pos_x' in col:
            X_norm[col] = normalize(X_norm[col], POS_MIN_X, POS_MAX_X)
        elif 'pos_y' in col or 'ball_pos_y' in col:
            X_norm[col] = normalize(X_norm[col], POS_MIN_Y, POS_MAX_Y)
        elif 'pos_z' in col or 'ball_pos_z' in col:
            X_norm[col] = normalize(X_norm[col], POS_MIN_Z, POS_MAX_Z)
        elif 'vel' in col and 'ball' not in col:
            X_norm[col] = normalize(X_norm[col], VEL_MIN, VEL_MAX)
        elif 'ball_vel' in col:
            X_norm[col] = normalize(X_norm[col], BALL_VEL_MIN, BALL_VEL_MAX)
        elif 'boost_amount' in col:
            X_norm[col] = normalize(X_norm[col], BOOST_MIN, BOOST_MAX)
        elif 'dist_to_ball' in col:
            X_norm[col] = normalize(X_norm[col], DIST_MIN, DIST_MAX)
        elif 'boost_pad' in col:
            X_norm[col] = normalize(X_norm[col], BOOST_PAD_MIN, BOOST_PAD_MAX)
        elif 'seconds_remaining' in col:
            # First, cap the values at 300 seconds, then normalize
            X_norm[col] = normalize(np.minimum(X_norm[col], 300.0), 0, 300)
        
        # Note: 'forward_x/y/z', 'team', and 'alive' are intentionally not normalized,
        # which matches the logic in your training scripts.

    # Return the data as a NumPy array, which is what PyTorch expects
    return X_norm.to_numpy()

# ====================== ARGUMENT PARSER ======================
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Augment a statistical ledger with ensemble model prediction statistics.")
    parser.add_argument('--data-dir', type=str, required=True, 
                        help='Path to the directory containing the dataset splits.')
    parser.add_argument('--output-csv', type=str, default='./augmented_statistical_ledger.csv', 
                        help='Path to save the final augmented CSV file.')
    parser.add_argument('--model-path-mlp', type=str, required=True, 
                        help='Path to the saved MLP model (.pth file).')
    parser.add_argument('--model-path-xgb-orange', type=str, required=True, 
                        help='Path to the saved ORANGE XGBoost model (.json file).')
    parser.add_argument('--model-path-xgb-blue', type=str, required=True, 
                        help='Path to the saved BLUE XGBoost model (.json file).')
    parser.add_argument('--parcel-size', type=int, default=512, 
                        help='The size of the grid parcels for discretization.')
    parser.add_argument('--filter-no-goal-tokens', action='store_true',
                        help='If set, only tokens where at least one goal was scored will be saved.')
    parser.add_argument('--min-prob', type=float, default=0.0,
                        help='Lower bound for goal probability to include a token (e.g., 0.3).')
    parser.add_argument('--max-prob', type=float, default=1.0,
                        help='Upper bound for goal probability to include a token (e.g., 0.7).')
    parser.add_argument('--exclude-player-z', action='store_true',
                        help='Exclude player Z-axis from token creation.')
    parser.add_argument('--exclude-ball-z', action='store_true',
                        help='Exclude ball Z-axis from token creation.')
    
    return parser.parse_args()

# ====================== HELPER FUNCTIONS ======================
def load_all_data(list_of_csv_paths):
    """
    Loads data from CSVs and returns three versions: a raw DataFrame for tokenization,
    raw features for XGBoost, normalized features for MLP, and the labels.
    """
    print(f"--- Loading data from {len(list_of_csv_paths)} files... ---")
    df_list = [pd.read_csv(path) for path in tqdm(list_of_csv_paths)]
    if not df_list:
        return None, None, None, None, None
        
    full_df = pd.concat(df_list, ignore_index=True).dropna()
    if full_df.empty:
        return None, None, None, None, None

    # --- 1. Get RAW features for XGBoost and tokenization ---
    feature_cols = [col for col in full_df.columns if col.startswith('p') or col.startswith('ball') or col.startswith('boost') or col == 'seconds_remaining']
    X_raw = full_df[feature_cols].copy()

    # --- 2. Create NORMALIZED features for MLP ---
    print("--- Normalizing features for MLP... ---")
    X_norm = full_df[feature_cols].copy() # Start with a fresh copy
    
    POS_MIN_X, POS_MAX_X = -4096, 4096; POS_MIN_Y, POS_MAX_Y = -6000, 6000; POS_MIN_Z, POS_MAX_Z = 0, 2044
    VEL_MIN, VEL_MAX = -2300, 2300; BOOST_MIN, BOOST_MAX = 0, 100; BALL_VEL_MIN, BALL_VEL_MAX = -6000, 6000
    BOOST_PAD_MIN, BOOST_PAD_MAX = 0, 10; DIST_MIN, DIST_MAX = 0, (8192**2 + 10240**2 + 2044**2)**0.5
    def normalize(val, min_val, max_val): return (val - min_val) / (max_val - min_val + 1e-8)

    for col in tqdm(X_norm.columns, desc="Normalizing columns"):
        if 'pos_x' in col or 'ball_pos_x' in col: X_norm[col] = normalize(X_norm[col], POS_MIN_X, POS_MAX_X)
        elif 'pos_y' in col or 'ball_pos_y' in col: X_norm[col] = normalize(X_norm[col], POS_MIN_Y, POS_MAX_Y)
        elif 'pos_z' in col or 'ball_pos_z' in col: X_norm[col] = normalize(X_norm[col], POS_MIN_Z, POS_MAX_Z)
        elif 'vel' in col and 'ball' not in col: X_norm[col] = normalize(X_norm[col], VEL_MIN, VEL_MAX)
        elif 'ball_vel' in col: X_norm[col] = normalize(X_norm[col], BALL_VEL_MIN, BALL_VEL_MAX)
        elif 'boost_amount' in col: X_norm[col] = normalize(X_norm[col], BOOST_MIN, BOOST_MAX)
        elif 'dist_to_ball' in col: X_norm[col] = normalize(X_norm[col], DIST_MIN, DIST_MAX)
        elif 'boost_pad' in col: X_norm[col] = normalize(X_norm[col], BOOST_PAD_MIN, BOOST_PAD_MAX)
        elif 'seconds_remaining' in col: X_norm[col] = normalize(np.minimum(X_norm[col], 300.0), 0, 300)

    # --- 3. Get labels ---
    y_orange = full_df['team_1_goal_in_event_window']
    y_blue = full_df['team_0_goal_in_event_window']
    
    return X_raw, X_norm.to_numpy(), y_orange, y_blue, full_df

def load_all_models(args, device):
    """Loads the trained MLP and XGBoost models from the paths provided."""
    print("\n--- Loading All Trained Models ---")
    
    # --- Load MLP Model ---
    try:
        model_mlp = BaselineMLP().to(device)
        print(f"  Loading MLP model from: {args.model_path_mlp}")
        checkpoint = torch.load(args.model_path_mlp, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            model_mlp.load_state_dict(checkpoint['model_state'])
        else:
            model_mlp.load_state_dict(checkpoint)
        model_mlp.eval()
        print("  MLP model loaded successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load the MLP model. Check the path. Error: {e}")
        return None, None, None

    # --- Load Orange XGBoost Model ---
    try:
        model_xgb_orange = xgb.XGBClassifier()
        print(f"  Loading XGBoost Orange model from: {args.model_path_xgb_orange}")
        model_xgb_orange.load_model(args.model_path_xgb_orange)
        print("  XGBoost Orange model loaded successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load the XGBoost Orange model. Check the path. Error: {e}")
        return None, None, None

    # --- Load Blue XGBoost Model ---
    try:
        model_xgb_blue = xgb.XGBClassifier()
        print(f"  Loading XGBoost Blue model from: {args.model_path_xgb_blue}")
        model_xgb_blue.load_model(args.model_path_xgb_blue)
        print("  XGBoost Blue model loaded successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load the XGBoost Blue model. Check the path. Error: {e}")
        return None, None, None
        
    return model_mlp, model_xgb_orange, model_xgb_blue

# ====================== CORE TOKENIZATION FUNCTIONS ======================
def discretize_position(x, y, z, parcel_size, exclude_z=False):
    """Converts a continuous 3D position into a discrete parcel tuple."""
    # --- UPDATE THESE LINES to use the passed-in argument ---
    px = int(np.floor(x / parcel_size))
    py = int(np.floor(y / parcel_size))
    if exclude_z:
        return (px, py)
    pz = int(np.floor(z / parcel_size))
    return (px, py, pz)

def create_token_from_row(row, args):
    """
    Creates a permutation-invariant token from a DataFrame row, respecting exclusion arguments.
    """
    try:
        # 1. Discretize Ball Position
        ball_parcel = discretize_position(
            float(row['ball_pos_x']), float(row['ball_pos_y']), float(row['ball_pos_z']),
            parcel_size=args.parcel_size, # <-- PASS THE ARGUMENT HERE
            exclude_z=args.exclude_ball_z
        )

        # 2. Discretize and Separate Player Positions by Team
        blue_positions = []; orange_positions = []
        for i in range(NUM_PLAYERS):
            player_pos = discretize_position(
                float(row[f'p{i}_pos_x']), float(row[f'p{i}_pos_y']), float(row[f'p{i}_pos_z']),
                parcel_size=args.parcel_size, # <-- AND PASS THE ARGUMENT HERE
                exclude_z=args.exclude_player_z
            )
            if i < 3: blue_positions.append(player_pos)
            else: orange_positions.append(player_pos)

        # 3. Sort Player Positions for Permutation Invariance
        blue_positions.sort()
        orange_positions.sort()

        # 4. Combine everything into a single, flat, readable string
        flat_blue = [str(coord) for pos_tuple in blue_positions for coord in pos_tuple]
        flat_orange = [str(coord) for pos_tuple in orange_positions for coord in pos_tuple]
        
        token_parts = [
            'B', *[str(c) for c in ball_parcel],
            'P_BLUE', *flat_blue,
            'P_ORANGE', *flat_orange
        ]
        return "_".join(token_parts)

    except (ValueError, KeyError):
        return None

# ====================== MAIN EXECUTION ======================
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Load Models ---
    model_mlp, model_xgb_orange, model_xgb_blue = load_all_models(args, device)
    if model_mlp is None: return

    # --- 2. Get All Predictions (Chunk by Chunk) ---
    print("\n--- Getting Predictions from All Models (Memory-Efficiently) ---")
    train_dir = os.path.join(args.data_dir, 'train')
    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.csv')]
    
    all_ensemble_probs_orange = []
    all_ensemble_probs_blue = []

    for file_path in tqdm(train_files, desc="Processing Chunks for Predictions"):
        df_chunk = pd.read_csv(file_path).dropna()
        if df_chunk.empty: continue
        
        # Prepare data for both models
        feature_cols = [col for col in df_chunk.columns if col.startswith('p') or col.startswith('ball') or col.startswith('boost') or col == 'seconds_remaining']
        X_raw = df_chunk[feature_cols]
        X_norm_np = normalize_dataframe_for_mlp(X_raw) # Normalize a copy

        # Get MLP predictions for the chunk
        with torch.no_grad():
            features_tensor = torch.tensor(X_norm_np, dtype=torch.float32).to(device)
            mlp_orange_prob, mlp_blue_prob = model_mlp(features_tensor)
        
        # Get XGBoost predictions for the chunk
        xgb_orange_prob = model_xgb_orange.predict_proba(X_raw)[:, 1]
        xgb_blue_prob = model_xgb_blue.predict_proba(X_raw)[:, 1]

        # Ensemble and store
        all_ensemble_probs_orange.extend((mlp_orange_prob.cpu().numpy().flatten() + xgb_orange_prob) / 2.0)
        all_ensemble_probs_blue.extend((mlp_blue_prob.cpu().numpy().flatten() + xgb_blue_prob) / 2.0)

    # Convert final lists to NumPy arrays
    final_ensemble_probs_orange = np.array(all_ensemble_probs_orange)
    final_ensemble_probs_blue = np.array(all_ensemble_probs_blue)

    # --- 3. Create Tokens and Aggregate Data (Memory-Efficiently) ---
    print("\n--- Creating Tokens and Aggregating Predictions ---")
    token_data = defaultdict(lambda: {'blue_goals': 0, 'blue_no_goals': 0, 'orange_goals': 0, 'orange_no_goals': 0, 'blue_probs': [], 'orange_probs': []})
    
    current_row_index = 0
    for file_path in tqdm(train_files, desc="Tokenizing and Aggregating"):
        # We read the file again, line by line, which is very fast
        with open(file_path, 'r') as f:
            header = f.readline().strip().split(',')
            for line in f:
                row_dict = dict(zip(header, line.strip().split(',')))
                token = create_token_from_row(row_dict, args)
                if token is None: continue

                token_data[token]['orange_probs'].append(final_ensemble_probs_orange[current_row_index])
                token_data[token]['blue_probs'].append(final_ensemble_probs_blue[current_row_index])
                
                if int(float(row_dict['team_1_goal_in_event_window'])) == 1: token_data[token]['orange_goals'] += 1
                else: token_data[token]['orange_no_goals'] += 1
                if int(float(row_dict['team_0_goal_in_event_window'])) == 1: token_data[token]['blue_goals'] += 1
                else: token_data[token]['blue_no_goals'] += 1
                
                current_row_index += 1

    # --- 5. Calculate Final Statistics and Save ---
    print("\n--- Calculating Final Statistics for each Token ---")
    output_data = []
    for token, data in tqdm(token_data.items(), desc="Calculating Stats"):
        blue_goals, blue_no_goals = data['blue_goals'], data['blue_no_goals']
        orange_goals, orange_no_goals = data['orange_goals'], data['orange_no_goals']
        blue_probs_np = np.array(data['blue_probs']); orange_probs_np = np.array(data['orange_probs'])
        
        output_data.append({
            'token': token,
            'blue_goals': blue_goals, 'blue_no_goals': blue_no_goals,
            'orange_goals': orange_goals, 'orange_no_goals': orange_no_goals,
            'total_occurrences': blue_goals + blue_no_goals,
            'blue_goal_prob_actual': blue_goals / (blue_goals + blue_no_goals) if (blue_goals + blue_no_goals) > 0 else 0,
            'orange_goal_prob_actual': orange_goals / (orange_goals + orange_no_goals) if (orange_goals + orange_no_goals) > 0 else 0,
            'ensemble_prob_blue_avg': np.mean(blue_probs_np),
            'ensemble_prob_blue_std': np.std(blue_probs_np),
            'ensemble_prob_blue_min': np.min(blue_probs_np),
            'ensemble_prob_blue_max': np.max(blue_probs_np),
            'ensemble_prob_orange_avg': np.mean(orange_probs_np),
            'ensemble_prob_orange_std': np.std(orange_probs_np),
            'ensemble_prob_orange_min': np.min(orange_probs_np),
            'ensemble_prob_orange_max': np.max(orange_probs_np)
        })

    # --- FILTERING LOGIC ---
    if args.filter_no_goal_tokens:
        print("--- Filtering out tokens with zero goals for both teams... ---")
        output_data = [d for d in output_data if d['blue_goals'] > 0 or d['orange_goals'] > 0]
        print(f"  Kept {len(output_data)} tokens after no-goal filtering.")

    if args.min_prob > 0.0 or args.max_prob < 1.0:
        print(f"--- Filtering tokens by ACTUAL probability range [{args.min_prob}, {args.max_prob}]... ---")
        filtered_data = []
        for record in output_data:
            prob_b = record['blue_goal_prob_actual'] # <-- FIX
            prob_o = record['orange_goal_prob_actual'] # <-- FIX
            if (prob_b >= args.min_prob and prob_b <= args.max_prob) or \
            (prob_o >= args.min_prob and prob_o <= args.max_prob):
                filtered_data.append(record)
        output_data = filtered_data
        print(f"  Kept {len(output_data)} tokens after probability filtering.")

    if not output_data:
        print("\n--- No tokens to save after processing and filtering. Exiting. ---")
        return

    output_df = pd.DataFrame(output_data)
    output_df = output_df.sort_values(by='total_occurrences', ascending=False).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    output_df.to_csv(args.output_csv, index=False)

    print(f"\n--- Statistical Ledger Creation Complete ---")
    print(f"Saved {len(output_df)} unique tokens.")
    print(f"Ledger saved to: {args.output_csv}")

if __name__ == '__main__':
    main()