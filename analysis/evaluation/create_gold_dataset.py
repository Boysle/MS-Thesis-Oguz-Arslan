import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize


"""
create_gold_dataset.py

Summary:
This script identifies "golden" (hard-to-predict) game-state tokens by building a
statistical ledger over discrete spatial tokens and selecting tokens whose goal
probability (for either team) falls within a configurable probability window.

Inputs:
- --data-dir: path to dataset root containing `train/`, `val/`, and `test/` subfolders

Outputs:
- single CSV file containing all rows (from the requested split(s)) whose token
    was classified as "golden" (--output-csv)
- optional diagnostic plots written to `ball_pos_plots` or `ball_pos_plots_no_kickoff`

Behavior / Flags:
- By default the script computes the ledger from all splits (train/val/test),
    then filters rows for the final golden dataset from the `test/` split. Use
    `--split-mode` to override and produce goldsets from `train`, `all`, or `test`.
- Use `--exclude-player-z` and/or `--exclude-ball-z` to drop Z when tokenizing.
- `--parcel-size` controls the spatial discretization used to form tokens.

Notes:
- The tokenization functions must match the ledger creation logic used elsewhere
    (e.g., `create_statistical_ledger.py`) so tokens are comparable across scripts.
"""




# ====================== CONFIGURATION & CONSTANTS ======================
PARCEL_SIZE = 1024
NUM_PLAYERS = 6

# ====================== ARGUMENT PARSER ======================
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Create a 'Gold Dataset' of hard-to-determine game states.")
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to the parent directory of the split dataset (containing train, val, test folders).')
    parser.add_argument('--output-csv', type=str, required=True,
                        help='Path to save the final, single "gold_dataset.csv" file.')
    
    parser.add_argument('--min-prob', type=float, default=0.1,
                        help='Lower bound for goal probability to be considered a "golden" token (default: 0.3).')
    parser.add_argument('--max-prob', type=float, default=0.9,
                        help='Upper bound for goal probability to be considered a "golden" token (default: 0.7).')
                        
    parser.add_argument('--parcel-size', type=int, default=1024,
                        help='The size of the grid parcels for discretization (must match ledger creation).')
    parser.add_argument('--heatmap-bins', type=int, default=128,
                    help='Number of bins per axis for the ball position heatmap (default: 128).')
    parser.add_argument('--exclude-player-z', action='store_true',
                        help='Exclude player Z-axis from token creation.')
    parser.add_argument('--exclude-ball-z', action='store_true',
                        help='Exclude ball Z-axis from token creation.')
    parser.add_argument('--exclude-kickoff', action='store_true',
                    help='Exclude rows before the first ball hit (ball_hit_team_num == 0.5).')

                        
    parser.add_argument('--split-mode', type=str, choices=['test', 'all', 'train'], default='test',
                        help='Which split(s) to use when creating the golden dataset: "test" (default), "train" or "all"')

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

def plot_ball_histogram(df, parcel_size, exclude_z=False, title="3D Histogram", save_path=None):
    # Discretize ball positions into parcel indices
    bx = np.floor(df['ball_pos_x'] / parcel_size).astype(int)
    by = np.floor(df['ball_pos_y'] / parcel_size).astype(int)
    if not exclude_z:
        bz = np.floor(df['ball_pos_z'] / parcel_size).astype(int)
    else:
        bz = np.zeros_like(bx)

    # Define bin edges aligned with parcel indices (+1 so the last edge is included)
    x_edges = np.arange(bx.min(), bx.max() + 2)
    y_edges = np.arange(by.min(), by.max() + 2)
    z_edges = np.arange(bz.min(), bz.max() + 2)

    # Histogram in parcel space
    hist, edges = np.histogramdd((bx, by, bz), bins=(x_edges, y_edges, z_edges))

    # Grid of bar positions
    xpos, ypos, zpos = np.meshgrid(
        edges[0][:-1],
        edges[1][:-1],
        edges[2][:-1],
        indexing="ij"
    )
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = zpos.ravel()

    dx = dy = 1  # one parcel per bin in X and Y
    dz = hist.ravel()  # counts per bar

    # Only plot non-empty bins
    mask = dz > 0
    xpos, ypos, zpos, dz = xpos[mask], ypos[mask], zpos[mask], dz[mask]

    # Colorize by height
    norm = Normalize(vmin=dz.min(), vmax=dz.max())
    colors = cm.viridis(norm(dz))

    # Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, shade=True)

    # Labels
    ax.set_xlabel(f"Ball X Parcel (size {parcel_size})")
    ax.set_ylabel(f"Ball Y Parcel (size {parcel_size})")
    ax.set_zlabel("Count")
    plt.title(title)

    # Square XY aspect (works in newer Matplotlib; fallback otherwise)
    try:
        ax.set_box_aspect([len(x_edges), len(y_edges), len(z_edges)])
    except Exception:
        pass

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()

def plot_ball_heatmap(df, bins, title="2D Heatmap of Ball Positions", save_path=None):
    bx = df['ball_pos_x'].to_numpy()
    by = df['ball_pos_y'].to_numpy()

    hist, x_edges, y_edges = np.histogram2d(bx, by, bins=bins)

    plt.figure(figsize=(10, 8))
    # Flatten histogram to get all counts
    flat_hist = hist.ravel()
    if len(flat_hist) > 1:
        vmax_cap = np.sort(flat_hist)[-2]
    else:
        vmax_cap = flat_hist.max()

    plt.imshow(hist.T, origin="lower", cmap="viridis",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            aspect="equal",
            vmin=0, vmax=vmax_cap)

    plt.colorbar(label="Count")
    plt.xlabel("Ball X Position")
    plt.ylabel("Ball Y Position")
    plt.title(title)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ====================== MAIN EXECUTION ======================
def main():
    args = parse_args()
    # Workflow summary (high-level):
    # 1) PASS 1 - Build a statistical ledger from the dataset splits. This creates
    #    aggregate goal/no-goal counts per discrete token so we can compute goal
    #    probabilities for each token.
    # 2) Identify golden tokens: tokens where EITHER team's goal probability falls
    #    in the configured [--min-prob, --max-prob] window.
    # 3) PASS 2 - Filter raw rows to produce the final golden dataset. Which
    #    split(s) are used for this filtering is controlled via --split-mode and
    #    can be `test` (default), `train`, or `all` (train+val+test).
    # 4) Save the concatenated golden dataset CSV and write diagnostic plots.
    # Edge cases handled: missing split directories, empty CSV lists, and
    # optional exclusion of kickoff rows via --exclude-kickoff.
    
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
        if args.exclude_kickoff:
            df = df[df['ball_hit_team_num'] != 0.5]
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
        
        blue_goals, blue_no_goals, orange_goals, orange_no_goals = counts
        total_blue = blue_goals + blue_no_goals
        total_orange = orange_goals + orange_no_goals

        blue_prob = blue_goals / total_blue if total_blue > 0 else 0
        orange_prob = orange_goals / total_orange if total_orange > 0 else 0

        # Keep the token if EITHER team's probability falls in the desired range
        if (args.min_prob <= blue_prob <= args.max_prob) or (args.min_prob <= orange_prob <= args.max_prob):
            golden_tokens.add(token)

    print(f"--- Token Statistics ---")
    print(f"Total unique tokens found: {len(token_counts)}")
    print(f"Golden tokens selected: {len(golden_tokens)}")

    if not golden_tokens:
        print("--- No tokens matched the criteria. No golden dataset will be created. ---")
        return

    # --- Collect all raw data before filtering for comparison ---
    print("\n--- Collecting full dataset for histogram plotting ---")
    all_dfs = []
    for file_path in tqdm(all_csv_files, desc="Loading all files for histogram"):
        df = pd.read_csv(file_path)
        all_dfs.append(df)
    all_data_df = pd.concat(all_dfs, ignore_index=True)

    # Decide which split(s) to use for creating the final golden dataset
    golden_dfs = []

    if args.split_mode == 'all':
        print("\n--- PASS 2 of 2: Filtering ALL splits (train, val, test) to create golden dataset... ---")
        target_files = all_csv_files
        desc = "Filtering all files for golden rows"
    elif args.split_mode == 'train':
        print("\n--- PASS 2 of 2: Filtering only the TRAIN split to create golden dataset... ---")
        train_dir = os.path.join(args.data_dir, 'train')
        if not os.path.isdir(train_dir):
            print(f"ERROR: train directory not found at {train_dir}")
            return
        target_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.csv')]
        desc = "Filtering train files for golden rows"
    else:
        print("\n--- PASS 2 of 2: Filtering only the TEST split to create golden dataset... ---")
        test_dir = os.path.join(args.data_dir, 'test')
        if not os.path.isdir(test_dir):
            print(f"ERROR: test directory not found at {test_dir}")
            return
        target_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.csv')]
        desc = "Filtering test files for golden rows"

    if not target_files:
        print("--- No CSV files found for the requested split(s). ---")
        return

    for file_path in tqdm(target_files, desc=desc):
        df = pd.read_csv(file_path)
        if args.exclude_kickoff:
            df = df[df['ball_hit_team_num'] != 0.5]
        df['token'] = df.apply(lambda row: create_token_from_row(row, args), axis=1)
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
    print(f"Average rows per golden token: {len(final_golden_df) / len(golden_tokens):.2f}")
    print(f"Saved to: {args.output_csv}")

    # Ensure plot folder exists
    plot_dir = "ball_pos_plots_no_kickoff" if args.exclude_kickoff else "ball_pos_plots"
    os.makedirs(plot_dir, exist_ok=True)

    # 3D histograms (parcel-based)
    plot_ball_histogram(all_data_df, args.parcel_size, args.exclude_ball_z,
                        title="Ball Position Histogram (All Data)",
                        save_path=os.path.join(plot_dir, "all_data_hist3d.png"))

    plot_ball_histogram(final_golden_df, args.parcel_size, args.exclude_ball_z,
                        title="Ball Position Histogram (Golden Dataset)",
                        save_path=os.path.join(plot_dir, "golden_data_hist3d.png"))

    # 2D density heatmaps (custom bins)
    plot_ball_heatmap(all_data_df, args.heatmap_bins,
                    title=f"Ball Position Heatmap (All Data, {args.heatmap_bins} bins)",
                    save_path=os.path.join(plot_dir, f"all_data_heatmap_{args.heatmap_bins}.png"))

    plot_ball_heatmap(final_golden_df, args.heatmap_bins,
                    title=f"Ball Position Heatmap (Golden Dataset, {args.heatmap_bins} bins)",
                    save_path=os.path.join(plot_dir, f"golden_data_heatmap_{args.heatmap_bins}.png"))

if __name__ == '__main__':
    main()