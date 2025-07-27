import os
import pandas as pd
import numpy as np
import shutil
import collections
from pathlib import Path

# --- Configuration ---
# Set to the specific paths you provided
INPUT_DIR = Path("E:\\Raw RL Esports Replays\\Day 3 Swiss Stage\\Round 1\\dataset_cleaned_v2")
OUTPUT_DIR = INPUT_DIR.parent / "split_dataset"
REPLAY_ID_COLUMN = "replay_id"  # The name of the column containing the game/replay ID

# Define the number of output chunks for each split
TRAIN_CHUNKS = 14
VAL_CHUNKS = 3
TEST_CHUNKS = 3

# Define the split ratios (used for the initial game ID split)
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# --- Main Script ---

def find_unique_games(input_path, id_column):
    """
    Scans all CSV files to find the complete set of unique replay/game IDs.
    """
    print(f"Scanning for unique games in '{input_path}'...")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input directory not found: {input_path}")
        
    all_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.csv')]
    if not all_files:
        raise ValueError(f"No CSV files found in '{input_path}'.")

    unique_game_ids = set()
    for file_path in all_files:
        try:
            df_chunk = pd.read_csv(file_path, usecols=[id_column])
            unique_game_ids.update(df_chunk[id_column].unique())
        except Exception as e:
            print(f"Warning: Could not process file {file_path}. Error: {e}")
            
    print(f"Found {len(unique_game_ids)} unique games.")
    return list(unique_game_ids)

def create_game_to_chunk_map(game_ids, train_r, val_r, test_r, train_c, val_c, test_c):
    """
    Performs a two-level split:
    1. Splits all game IDs into main train/val/test groups.
    2. Subdivides each group into the desired number of chunks.
    3. Creates a final mapping from each game ID to its exact output file (e.g., ('train', 5)).
    """
    print("Randomly shuffling and splitting game IDs into chunks...")
    
    # 1. Initial split into main train/val/test groups
    np.random.shuffle(game_ids)
    total_games = len(game_ids)
    train_end = int(total_games * train_r)
    val_end = train_end + int(total_games * val_r)
    
    train_ids = game_ids[:train_end]
    val_ids = game_ids[train_end:val_end]
    test_ids = game_ids[val_end:]
    
    print(f"Main split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test games.")

    # 2. Subdivide each group's game list into the desired number of chunks
    subdivided_train_ids = np.array_split(train_ids, train_c)
    subdivided_val_ids = np.array_split(val_ids, val_c)
    subdivided_test_ids = np.array_split(test_ids, test_c)
    
    # 3. Create the final mapping from a game ID to its destination chunk
    game_to_chunk_map = {}
    for i, chunk_list in enumerate(subdivided_train_ids):
        for game_id in chunk_list:
            game_to_chunk_map[game_id] = ('train', i)
            
    for i, chunk_list in enumerate(subdivided_val_ids):
        for game_id in chunk_list:
            game_to_chunk_map[game_id] = ('val', i)
            
    for i, chunk_list in enumerate(subdivided_test_ids):
        for game_id in chunk_list:
            game_to_chunk_map[game_id] = ('test', i)

    print("Created mapping from each game ID to its final destination chunk.")
    return game_to_chunk_map

def distribute_data_to_chunks(input_path, output_path, id_column, game_map):
    """
    Reads all source CSVs and distributes rows into the correct
    output CHUNK file based on the game-to-chunk mapping.
    """
    # Prepare output directories
    for split_name in ['train', 'val', 'test']:
        split_dir = os.path.join(output_path, split_name)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(split_dir)
        
    print(f"\nWriting data to sub-folders in '{output_path}'...")
    all_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.csv')]

    # Process each source chunk file
    for i, file_path in enumerate(all_files):
        print(f"Processing source chunk {i+1}/{len(all_files)}: {os.path.basename(file_path)}")
        df = pd.read_csv(file_path)
        
        # Add a temporary column to know where each row should go (e.g., ('train', 5))
        df['split_group'] = df[id_column].map(game_map)
        
        # Group by the destination and append to the correct output file
        for split_group, group_df in df.groupby('split_group'):
            if pd.isna(split_group):
                print(f"Warning: Found rows with no assigned split group. These will be skipped.")
                continue

            # split_group will be a tuple like ('train', 5)
            split_name, chunk_index = split_group
            
            # Define the output directory and file name
            output_dir = os.path.join(output_path, split_name)
            output_file = os.path.join(output_dir, f"{split_name}_chunk_{chunk_index}.csv")
            
            # Drop the temporary column before saving
            group_df = group_df.drop(columns=['split_group'])
            
            # Write to the specific chunk CSV file
            group_df.to_csv(
                output_file, 
                mode='a', 
                header=not os.path.exists(output_file), 
                index=False
            )
            
    print("\nDataset splitting process completed successfully!")
    print(f"Output generated in '{OUTPUT_DIR}'. You will find {TRAIN_CHUNKS} train, {VAL_CHUNKS} val, and {TEST_CHUNKS} test files.")


if __name__ == '__main__':
    # --- Execute the pipeline ---
    # 1. Get all unique game IDs from the entire dataset
    unique_ids = find_unique_games(INPUT_DIR, REPLAY_ID_COLUMN)
    
    # 2. Create the detailed map from each game ID to a specific output chunk file
    game_id_map = create_game_to_chunk_map(
        unique_ids,
        TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO,
        TRAIN_CHUNKS, VAL_CHUNKS, TEST_CHUNKS
    )
    
    # 3. Read the data chunk by chunk and write rows to the correct final chunk files
    distribute_data_to_chunks(
        INPUT_DIR, OUTPUT_DIR, REPLAY_ID_COLUMN, game_id_map
    )