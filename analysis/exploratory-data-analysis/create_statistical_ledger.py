import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# ====================== CONFIGURATION & CONSTANTS ======================
NUM_PLAYERS = 6

# ====================== ARGUMENT PARSER ======================
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Create a statistical ledger of game state tokens from replay data.")
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to the directory containing the dataset splits (e.g., .../split_dataset).')
    parser.add_argument('--output-csv', type=str, default='./statistical_ledger.csv',
                        help='Path to save the final output CSV file.')
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
    parser.add_argument('--parcel-size', type=int, default=512,
                        help='The size of the grid parcels for discretization (default: 512).')
    parser.add_argument('--split-mode', type=str, choices=['train', 'val', 'test', 'all'], default='train',
                        help='Which split(s) to process when building the ledger: "train" (default), "val", "test", or "all".')
                        
    return parser.parse_args()


# ====================== CORE TOKENIZATION FUNCTIONS (UPDATED) ======================
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
    
    # Determine which dataset split(s) to process
    if args.split_mode == 'all':
        splits = ['train', 'val', 'test']
    else:
        splits = [args.split_mode]

    all_csv_files = []
    for split in splits:
        split_dir = os.path.join(args.data_dir, split)
        if not os.path.isdir(split_dir):
            print(f"ERROR: '{split}' subfolder not found in the specified data directory: {args.data_dir}")
            return
        all_csv_files.extend([os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith('.csv')])

    if not all_csv_files:
        print(f"ERROR: No CSV files found in the requested split(s): {splits}")
        return

    token_counts = defaultdict(lambda: [0, 0, 0, 0])
    print(f"--- Found {len(all_csv_files)} CSV files to process in splits: {splits} ---")

    for file_path in tqdm(all_csv_files, desc="Processing files"):
        df = pd.read_csv(file_path)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing rows in {os.path.basename(file_path)}", leave=False):
            # Pass args object to the token creation function
            token = create_token_from_row(row, args)
            if token is None: continue
            
            is_blue_goal = int(row['team_0_goal_in_event_window'])
            is_orange_goal = int(row['team_1_goal_in_event_window'])

            if is_blue_goal == 1: token_counts[token][0] += 1
            else: token_counts[token][1] += 1
            if is_orange_goal == 1: token_counts[token][2] += 1
            else: token_counts[token][3] += 1
    
    print("\n--- Aggregating results and preparing final data ---")

    output_data = []
    # Use the potentially filtered dictionary now
    for token, counts in token_counts.items():
        blue_goals, blue_no_goals, orange_goals, orange_no_goals = counts
        total_blue = blue_goals + blue_no_goals
        total_orange = orange_goals + orange_no_goals
        
        output_data.append({
            'token': token,
            'blue_goals': blue_goals, 'blue_no_goals': blue_no_goals,
            'orange_goals': orange_goals, 'orange_no_goals': orange_no_goals,
            'total_occurrences': max(total_blue, total_orange),
            'blue_goal_prob': blue_goals / total_blue if total_blue > 0 else 0.0,
            'orange_goal_prob': orange_goals / total_orange if total_orange > 0 else 0.0
        })

    # --- NEW FILTERING LOGIC ---
    if args.filter_no_goal_tokens:
        print("--- Filtering out tokens with zero goals for both teams... ---")
        output_data = [d for d in output_data if d['blue_goals'] > 0 or d['orange_goals'] > 0]
        print(f"  Kept {len(output_data)} tokens after no-goal filtering.")

    if args.min_prob > 0.0 or args.max_prob < 1.0:
        print(f"--- Filtering tokens by probability range [{args.min_prob}, {args.max_prob}]... ---")
        filtered_data = []
        for record in output_data:
            prob_b = record['blue_goal_prob']
            prob_o = record['orange_goal_prob']
            # Keep the token if EITHER team's probability is in the desired range
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