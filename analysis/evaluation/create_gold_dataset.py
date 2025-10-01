import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import re

# ====================== CONFIGURATION & CONSTANTS ======================
PARCEL_SIZE = 512
NUM_PLAYERS = 6

# ====================== ARGUMENT PARSER ======================
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Create a 'Gold Dataset' of hard-to-determine game states.")
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to the parent directory of the split dataset (containing train, val, test folders).')
    parser.add_argument('--output-csv', type=str, required=True,
                        help='Path to save the final, single "gold_dataset.csv" file.')
    
    parser.add_argument('--min-prob', type=float, default=0.01,
                        help='Lower bound for goal probability to be considered a "golden" token (default: 0.3).')
    parser.add_argument('--max-prob', type=float, default=0.99,
                        help='Upper bound for goal probability to be considered a "golden" token (default: 0.7).')
                        
    parser.add_argument('--parcel-size', type=int, default=1024,
                        help='The size of the grid parcels for discretization (must match ledger creation).')
    parser.add_argument('--exclude-player-z', action='store_true',
                        help='Exclude player Z-axis from token creation.')
    parser.add_argument('--exclude-ball-z', action='store_true',
                        help='Exclude ball Z-axis from token creation.')
                        
    return parser.parse_args()

# ====================== CORE TOKENIZATION FUNCTIONS ======================
# These functions must be identical to the ones in create_statistical_ledger.py
def discretize_position(x, y, z, parcel_size, exclude_z=True):
    """Converts a continuous 3D position into a discrete parcel tuple."""
    px = int(np.floor(x / parcel_size))
    py = int(np.floor(y / parcel_size))
    if exclude_z:
        return (px, py)
    pz = int(np.floor(z / parcel_size))
    return (px, py, pz)

def create_token_from_row(row, args):
    """Creates a permutation-invariant token from a DataFrame row."""
    try:
        ball_parcel = discretize_position(
            float(row['ball_pos_x']), float(row['ball_pos_y']), float(row['ball_pos_z']),
            parcel_size=args.parcel_size, exclude_z=args.exclude_ball_z
        )
        blue_positions = []; orange_positions = []
        for i in range(NUM_PLAYERS):
            player_pos = discretize_position(
                float(row[f'p{i}_pos_x']), float(row[f'p{i}_pos_y']), float(row[f'p{i}_pos_z']),
                parcel_size=args.parcel_size, exclude_z=args.exclude_player_z
            )
            if i < 3: blue_positions.append(player_pos)
            else: orange_positions.append(player_pos)
        
        blue_positions.sort(); orange_positions.sort()
        
        flat_blue = [str(coord) for pos_tuple in blue_positions for coord in pos_tuple]
        flat_orange = [str(coord) for pos_tuple in orange_positions for coord in pos_tuple]
        token_parts = ['B', *[str(c) for c in ball_parcel], 'P_BLUE', *flat_blue, 'P_ORANGE', *flat_orange]
        return "_".join(token_parts)
    except (ValueError, KeyError):
        return None

# ====================== MAIN EXECUTION ======================
def main():
    args = parse_args()
    
    # --- 1. Find all CSV files across train, val, and test splits ---
    all_csv_files = []
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(args.data_dir, split)
        if os.path.isdir(split_dir):
            all_csv_files.extend([os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith('.csv')])
    
    if not all_csv_files:
        print(f"ERROR: No CSV files found in the subdirectories of {args.data_dir}")
        return

    # --- PASS 1: Build the complete statistical ledger in memory ---
    print("--- PASS 1 of 2: Building statistical ledger from all data... ---")
    token_counts = defaultdict(lambda: [0, 0, 0, 0])
    for file_path in tqdm(all_csv_files, desc="Processing files for ledger"):
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            token = create_token_from_row(row, args)
            if token is None: continue
            
            if int(row['team_0_goal_in_event_window']) == 1: token_counts[token][0] += 1
            else: token_counts[token][1] += 1
            if int(row['team_1_goal_in_event_window']) == 1: token_counts[token][2] += 1
            else: token_counts[token][3] += 1
            
    # --- Identify the "Golden Tokens" ---
    print("\n--- Identifying 'Golden' Tokens based on probability range... ---")
    golden_tokens = set()
    for token, counts in token_counts.items():
        blue_goals, blue_no_goals, orange_goals, orange_no_goals = counts
        total_blue = blue_goals + blue_no_goals
        total_orange = orange_goals + orange_no_goals
        
        blue_prob = blue_goals / total_blue if total_blue > 0 else 0
        orange_prob = orange_goals / total_orange if total_orange > 0 else 0
        
        if (args.min_prob <= blue_prob <= args.max_prob) or \
           (args.min_prob <= orange_prob <= args.max_prob):
            golden_tokens.add(token)
            
    print(f"Found {len(golden_tokens)} unique 'golden' tokens out of {len(token_counts)} total tokens.")
    if not golden_tokens:
        print("--- No tokens matched the criteria. No golden dataset will be created. ---")
        return

    # --- PASS 2: Filter the original data to create the golden dataset ---
    print("\n--- PASS 2 of 2: Filtering data to create golden dataset... ---")
    golden_dfs = []
    for file_path in tqdm(all_csv_files, desc="Filtering files for golden rows"):
        df = pd.read_csv(file_path)
        # Vectorized token creation for the whole chunk is faster
        df['token'] = df.apply(lambda row: create_token_from_row(row, args), axis=1)
        # Filter rows where the token is in our golden set
        golden_chunk = df[df['token'].isin(golden_tokens)]
        golden_dfs.append(golden_chunk)

    if not golden_dfs:
        print("--- No matching rows found for the golden tokens. ---")
        return

    # --- Concatenate and Save the Final Dataset ---
    print("\n--- Concatenating and saving the final golden dataset... ---")
    final_golden_df = pd.concat(golden_dfs, ignore_index=True)
    # Drop the temporary token column to restore original structure
    final_golden_df = final_golden_df.drop(columns=['token'])
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    final_golden_df.to_csv(args.output_csv, index=False)

    print("\n--- Golden Dataset Creation Complete ---")
    print(f"Total rows in golden dataset: {len(final_golden_df)}")
    print(f"Saved to: {args.output_csv}")

if __name__ == '__main__':
    main()