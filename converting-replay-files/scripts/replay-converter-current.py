#!/usr/bin/env python3
"""
Rocket League Replay Processor

This script processes Rocket League replay files (.replay) using carball to extract:
- Player positions and velocities
- Ball tracking data
- Game events (goals)
- Team information

The processed data is downsampled, cleaned, and saved to CSV files with consistent player ordering.

Key Features:
- Parallel processing of multiple replays
- Data downsampling with event preservation
- Post-goal frame cleaning
- Comprehensive logging and error handling
- Class balance tracking for machine learning
"""

import subprocess
import pandas as pd
import json
from pathlib import Path
import os
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from threading import Lock
from dataclasses import dataclass
import io
import re
import itertools
from threading import Lock
# ==================================================================

pd.set_option('future.no_silent_downcasting', True)

# ==================================================================
# Global Configuration
# ==================================================================
# Using global variables for configuration that would typically come from a config file
SCRIPT_DIR = Path(__file__).resolve().parent
CARBALL_EXE = SCRIPT_DIR / "carball.exe"  # Path to carball executable
PARENT_DIR = Path(r"E:\\Raw RL Esports Replays\\Big Replay Dataset")  # Root directory containing replays
MAX_WORKERS = 4  # Maximum parallel threads for processings
POSITIVE_STATE_TARGET_HZ = 5 # Target sampling frequency in Hz for positive states
NEGATIVE_STATE_TARGET_HZ = 5 # Target sampling frequency in Hz for negative states
GOAL_ANTICIPATION_WINDOW_SECONDS = 5 # Time window before a goal to label as positive state
LOG_LEVEL = logging.INFO  # Logging verbosity
# Global variables for output filenames and formats
CARBALL_OUTPUT_FOLDER_FORMAT = "Output-{stem}"  # Format for carball's temporary output folder
INDIVIDUAL_REPLAY_CSV_FILENAME_FORMAT = "Output-{stem}" # Placeholder, will be updated in main
INDIVIDUAL_CSVS_SUBFOLDER_NAME = "Processed_Individual_Replays" # Placeholder
COMBINED_DATASET_CSV_FILENAME = "all_replays_combined_dataset.csv" # Placeholder
PROCESSING_SUMMARY_FILENAME = "processing_summary.txt" # Placeholder
# Constants for null handling
MAX_MAP_DISTANCE = 13411.30
DEMO_POSITION = [0.0, 0.0, 0.0]  # Position to fill for demoed players
DEMO_QUATERNION = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion
DEMO_VELOCITY = [0.0, 0.0, 0.0]
NULL_HANDLING_RULES = {
    # Player columns
    'pos_[xyz]': {'action': 'fill', 'value': 0, 'alive_check': True},
    'vel_[xyz]': {'action': 'fill', 'value': 0, 'alive_check': True},
    'quat_[wxyz]': {'action': 'fill', 'value': 0, 'alive_check': True},
    'boost_amount': {'action': 'fill', 'value': 0, 'alive_check': False},
    'dist_to_ball': {'action': 'fill', 'value': MAX_MAP_DISTANCE, 'alive_check': False},
    'forward_[xyz]': {'action': 'fill', 'value': 0, 'alive_check': True},
    
    # Ball columns
    'ball_pos_[xyz]': {'action': 'drop'},
    'ball_vel_[xyz]': {'action': 'fill', 'value': 0},
    'ball_hit_team_num': {'action': 'fill', 'value': 0.5},
    
    # Game columns
    'seconds_remaining': {'action': 'fill', 'value': 0},
    'is_overtime': {'action': 'fill', 'value': False},
    
    # Event columns
    'team_[01]_goal_in_event_window': {'action': 'fill', 'value': 0},

    # Context columns
    'replay_id': {'action': 'fill', 'value': 'UNKNOWN'},
    'blue_score': {'action': 'fill', 'value': 0},
    'orange_score': {'action': 'fill', 'value': 0},
    'score_difference': {'action': 'fill', 'value': 0}
}

# Big boost pad positions (x, y, z)
BOOST_PAD_POSITIONS = {
    0: [-3072.0, -4096.0, 73.0],
    1: [3072.0, -4096.0, 73.0],
    2: [-3584.0, 0.0, 73.0],
    3: [3584.0, 0.0, 73.0],
    4: [-3072.0, 4096.0, 73.0],
    5: [3072.0, 4096.0, 73.0]
}
BOOST_RESPAWN_TIME = 10  # seconds

# Global counters for tracking class balance across all replays
total_counts = {"positive": 0, "negative": 0}
counts_lock = Lock()  # Thread lock for safe counter updates

# Fix for Windows console encoding
if sys.stdout.encoding != 'UTF-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Creating a thread-safe counter for generating unique replay IDs
replay_id_counter = itertools.count()
replay_id_lock = Lock()


# ==================================================================
# Data Structures
# ==================================================================
@dataclass
class ReplayStats:
    """Container for statistics about a processed replay"""
    replay_name: str
    rows: int
    processing_time: float
    success: bool
    error: Optional[str] = None

@dataclass
class GoalEvent:
    """Container for goal event information"""
    frame: int
    time: float
    team: int  # 0 for blue, 1 for orange

# ==================================================================
# Logging Configuration
# ==================================================================
def configure_logging():
    """Set up logging with both file and console output"""
    logging.basicConfig(
        level=LOG_LEVEL, # Use LOG_LEVEL from global config
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('replay_processor.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)  # Use the patched stdout
        ]
    )
    # Suppress verbose logging from dependencies
    logging.getLogger('parquet').setLevel(logging.WARNING)

# ==================================================================
# Core Processing Functions
# ==================================================================

def add_replay_id_column(df: pd.DataFrame, replay_id: str) -> pd.DataFrame:
    """Adds a column with a unique identifier for the replay."""
    df['replay_id'] = replay_id
    # It's good practice to make it a categorical type if you have many replays
    df['replay_id'] = df['replay_id'].astype('category')
    return df

def add_score_context_columns(df: pd.DataFrame, goal_events: List[GoalEvent]) -> pd.DataFrame:
    """
    Adds blue_score, orange_score, and score_difference columns based on goal events.

    Args:
        df: The DataFrame of game frames, must have 'original_frame'.
        goal_events: A list of GoalEvent objects, sorted by frame number.

    Returns:
        The DataFrame with added score context columns.
    """
    if df.empty or 'original_frame' not in df.columns:
        return df

    # Initialize score columns
    df['blue_score'] = 0
    df['orange_score'] = 0

    # Ensure goal events are sorted by frame to process them chronologically
    sorted_goals = sorted(goal_events, key=lambda g: g.frame)

    current_blue_score = 0
    current_orange_score = 0
    
    # Set the starting frame for score application
    last_frame_processed = -1

    for goal in sorted_goals:
        goal_frame = goal.frame
        
        # Apply the previous score state up to the frame of the current goal
        # The mask finds all rows with original_frame > last_frame_processed and <= goal_frame
        score_mask = (df['original_frame'] > last_frame_processed) & (df['original_frame'] <= goal_frame)
        df.loc[score_mask, 'blue_score'] = current_blue_score
        df.loc[score_mask, 'orange_score'] = current_orange_score

        # Update the score *after* the goal event
        if goal.team == 0: # Blue goal
            current_blue_score += 1
        else: # Orange goal
            current_orange_score += 1
            
        last_frame_processed = goal_frame

    # Apply the final score to all remaining frames after the last goal
    if last_frame_processed != -1:
        final_score_mask = df['original_frame'] > last_frame_processed
        df.loc[final_score_mask, 'blue_score'] = current_blue_score
        df.loc[final_score_mask, 'orange_score'] = current_orange_score

    # Calculate score_difference (Orange - Blue)
    df['score_difference'] = df['orange_score'] - df['blue_score']
    
    logging.info(f"Added score context. Final score: Blue {current_blue_score} - Orange {current_orange_score}")
    return df

def downsample_data(df: pd.DataFrame, 
                   original_hz: int = 30, 
                   event_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Downsample dataframe based on globally configured POSITIVE_STATE_TARGET_HZ 
    and NEGATIVE_STATE_TARGET_HZ.
    
    Positive states (events identified by event_columns) are sampled at POSITIVE_STATE_TARGET_HZ.
    Negative states (non-events) are sampled at NEGATIVE_STATE_TARGET_HZ.
    
    Args:
        df: Input DataFrame with time-series data.
        original_hz: Original sampling frequency (default 30Hz for Rocket League).
        event_columns: List of column names marking important events (positive states).
        
    Returns:
        Downsampled DataFrame.
    """
    if df.empty:
        return df

    if original_hz <= 0:
        logging.warning(f"Original HZ ({original_hz}) is non-positive, cannot downsample. Returning original DataFrame.")
        return df

    # Use global target frequencies
    # These are accessed directly from the global scope
    
    # Handle the simple case: if both target frequencies are the same
    if POSITIVE_STATE_TARGET_HZ == NEGATIVE_STATE_TARGET_HZ:
        target_hz_common = POSITIVE_STATE_TARGET_HZ
        if target_hz_common >= original_hz or target_hz_common <= 0:
            # No downsampling needed, or invalid target HZ for downsampling
            return df 
        
        sampling_interval = int(original_hz / target_hz_common)
        # sampling_interval will be >= 1 because target_hz_common < original_hz and target_hz_common > 0
        return df.iloc[::sampling_interval].reset_index(drop=True)

    # --- Proceed with potentially different HZ for positive and negative states ---
    
    # Calculate sampling intervals for positive and negative states
    # Interval is 1 if no downsampling for that category (target >= original or target <= 0)
    positive_interval = 1
    if 0 < POSITIVE_STATE_TARGET_HZ < original_hz:
        positive_interval = int(original_hz / POSITIVE_STATE_TARGET_HZ)
    
    negative_interval = 1
    if 0 < NEGATIVE_STATE_TARGET_HZ < original_hz:
        negative_interval = int(original_hz / NEGATIVE_STATE_TARGET_HZ)

    # If, after considering individual HZ values, both intervals are 1,
    # it means no actual downsampling is required for either category relative to original_hz.
    # (e.g. original=30, P_HZ=30, N_HZ=40 -> P_interval=1, N_interval=1)
    if positive_interval == 1 and negative_interval == 1:
        return df

    # Create mask for positive state rows
    positive_mask = pd.Series(False, index=df.index)
    if event_columns: 
        for col in event_columns:
            if col in df.columns: # Ensure column exists
                positive_mask = positive_mask | (df[col] == 1)
    
    final_keep_indices = set()

    # Get indices for positive states and sample them
    positive_indices = df.index[positive_mask]
    if not positive_indices.empty:
        final_keep_indices.update(positive_indices[::positive_interval])
    
    # Get indices for negative states and sample them
    negative_indices = df.index[~positive_mask]
    if not negative_indices.empty:
        final_keep_indices.update(negative_indices[::negative_interval])
        
    if not final_keep_indices:
        # This could happen if df was not empty, but after attempting to sample,
        # no indices were selected (e.g., intervals too large for the number of rows in categories).
        # Return an empty DataFrame with original columns.
        return df.iloc[[]].reset_index(drop=True) 

    sorted_indices = sorted(list(final_keep_indices))
    return df.loc[sorted_indices].reset_index(drop=True)

def get_empirical_tick_rate(game_df: pd.DataFrame, expected_rates: List[int]) -> int:
    """
    Determines the most likely tick rate by analyzing the modal 'delta' value.

    Args:
        game_df: The DataFrame loaded from __game.parquet, must contain a 'delta' column.
        expected_rates: A list of common tick rates to snap the calculated value to.

    Returns:
        The nearest expected tick rate (e.g., 30, 60, or 120). Defaults to 30.
    """
    if 'delta' not in game_df.columns or game_df['delta'].empty:
        logging.warning("Could not determine empirical tick rate: 'delta' column missing or empty. Defaulting to 30Hz.")
        return 30

    # The first delta is often 0 or unusual, so we look at the first couple of seconds of data.
    # Taking up to the first 300 frames is a safe bet to find a stable delta.
    sample_deltas = game_df['delta'].iloc[1:300].dropna()

    if sample_deltas.empty:
        logging.warning("Not enough valid delta values in the first 300 frames to determine tick rate. Defaulting to 30Hz.")
        return 30

    # Find the modal (most common) delta. This is more robust than the mean against lag spikes.
    # We round to 4 decimal places to group similar float values (e.g., 0.0166 and 0.0167).
    modal_delta = sample_deltas.round(4).mode()[0]

    if modal_delta <= 0:
        logging.warning(f"Modal delta is non-positive ({modal_delta:.4f}). Cannot calculate tick rate. Defaulting to 30Hz.")
        return 30

    # Calculate the raw frequency
    raw_hz = 1.0 / modal_delta
    
    # Snap the calculated frequency to the nearest expected rate
    # This corrects for small variations and gives a clean integer value.
    # It finds the rate in the list that has the minimum absolute difference from our raw_hz.
    snapped_hz = min(expected_rates, key=lambda rate: abs(rate - raw_hz))

    logging.info(f"Empirically determined tick rate: modal delta={modal_delta:.4f}s, raw_hz={raw_hz:.2f}Hz, snapped to {snapped_hz}Hz.")
    
    return snapped_hz

def adjust_seconds_remaining_for_overtime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjusts the 'seconds_remaining' column to handle overtime.
    In overtime, time counts up from 0, so this function makes it negative.

    This version is more robust and uses vectorized operations.

    Args:
        df: DataFrame containing 'is_overtime', 'seconds_remaining', and 'time' columns.

    Returns:
        The DataFrame with the adjusted 'seconds_remaining' column.
    """
    required_cols = {'is_overtime', 'seconds_remaining', 'time'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        logging.warning(f"adjust_seconds_remaining_for_overtime: Missing required columns: {missing}. Skipping adjustment.")
        return df

    # Create a mask for all rows that are in overtime
    overtime_mask = df['is_overtime']

    # If there's no overtime, do nothing.
    if not overtime_mask.any():
        logging.debug("No overtime detected, 'seconds_remaining' not adjusted.")
        return df

    # Find the game time at the very start of overtime.
    # We find the first index where the mask is True and get its 'time'.
    overtime_start_time = df.loc[overtime_mask.idxmax(), 'time']
    logging.info(f"Overtime detected. Start time: {overtime_start_time:.2f}s")
    
    # Calculate the new values for overtime rows
    time_elapsed_in_ot = df['time'] - overtime_start_time
    
    # Use np.where for a conditional assignment.
    # Condition: Is it overtime?
    # If True: use -time_elapsed_in_ot
    # If False: use the original 'seconds_remaining' value
    df['seconds_remaining'] = np.where(
        overtime_mask,
        -time_elapsed_in_ot,
        df['seconds_remaining']
    )

    # We round before casting to handle floats like -0.9 becoming 0 instead of -1.
    df['seconds_remaining'] = df['seconds_remaining'].round().astype('int64')

    return df



# Place this in Core Processing Functions, replacing the previous version

# Place this in Core Processing Functions, replacing the previous version


def calculate_distance(pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> float:
    """
    Calculate Euclidean distance between objects in 3D space.
    
    Args:
        pos1: Tuple of (x, y, z) object 1 coordinates
        pos2: Tuple of (x, y, z) object 2 coordinates
        
    Returns:
        Distance between objects in 3D space
    """
    return np.sqrt(
        (pos1[0] - pos2[0])**2 +
        (pos1[1] - pos2[1])**2 +
        (pos1[2] - pos2[2])**2
    )

def quaternion_to_forward_vector(qw: float, qx: float, qy: float, qz: float) -> Tuple[float, float, float]:
    """
    Convert Rocket League quaternion to proper forward unit vector.
    Corrected coordinate system handling:
    - Forward vector should point where the car is facing
    - RL uses left-handed Z-up coordinate system
    
    Args:
        qw, qx, qy, qz: Quaternion components (w, x, y, z)
        
    Returns:
        (forward_x, forward_y, forward_z) unit vector
    """
    # Correct forward vector calculation for RL's coordinate system
    x = 1 - 2 * (qy**2 + qz**2)
    y = 2 * (qx*qy + qw*qz)
    z = 2 * (qx*qz - qw*qy)
    
    # Normalize (should already be unit length, but ensure precision)
    norm = np.sqrt(x**2 + y**2 + z**2)
    if norm == 0: return (1.0, 0.0, 0.0) # Avoid division by zero, default to forward-X
    return (x/norm, y/norm, z/norm)

def find_nearest_boost_pad(player_pos: Tuple[float, float, float]) -> int:
    """Find closest boost pad using existing distance function"""
    closest_pad = None
    min_distance = float('inf')
    
    for pad_id, pad_pos in BOOST_PAD_POSITIONS.items():
        dist = calculate_distance(player_pos, pad_pos)
        if dist < min_distance:
            min_distance = dist
            closest_pad = pad_id
            
    return closest_pad # type: ignore

def update_boost_pad_timers(df: pd.DataFrame) -> pd.DataFrame:
    """100% guaranteed to create and populate boost pad columns"""
    # ===== 1. FORCE CREATE COLUMNS FIRST (More efficiently) =====
    pad_columns_to_init = {f'boost_pad_{i}_respawn': 0.0 for i in range(6)}
    df = df.assign(**pad_columns_to_init)
    
    # ===== 2. Initialize timers =====
    pad_timers = {i: 0.0 for i in range(6)}
    last_time = df.iloc[0]['time'] if len(df) > 0 else 0.0
    
    # ===== 3. Process each frame =====
    for idx, row in df.iterrows():
        current_time = row['time']
        time_delta = current_time - last_time if current_time > last_time else 0 # ensure non-negative delta
        
        # Update active timers
        for pad_id in range(6):
            if pad_timers[pad_id] > 0:
                pad_timers[pad_id] = max(0, pad_timers[pad_id] - time_delta)
            df.loc[idx, f'boost_pad_{pad_id}_respawn'] = pad_timers[pad_id]  # Direct assignment
        
        # Check for boost pickups
        for player_id_idx in range(6): # Iterate 0-5 for p0-p5
            pickup_col = f'p{player_id_idx}_boost_pickup'
            if pickup_col not in row or pd.isna(row[pickup_col]):
                continue
                
            if row[pickup_col] == 2:  # Big pad pickup (value 2 means big pad)
                try:
                    # Ensure player position columns exist for this player
                    pos_x_col, pos_y_col, pos_z_col = f'p{player_id_idx}_pos_x', f'p{player_id_idx}_pos_y', f'p{player_id_idx}_pos_z'
                    if not all(c in row and pd.notna(row[c]) for c in [pos_x_col, pos_y_col, pos_z_col]):
                        continue

                    player_pos = (
                        row[pos_x_col],
                        row[pos_y_col],
                        row[pos_z_col]
                    )
                    # Find nearest pad
                    nearest_pad_id = find_nearest_boost_pad(player_pos)
                    
                    if nearest_pad_id is not None:
                        if pad_timers[nearest_pad_id] < BOOST_RESPAWN_TIME - 0.1: 
                             pad_timers[nearest_pad_id] = BOOST_RESPAWN_TIME
                        df.loc[idx, f'boost_pad_{nearest_pad_id}_respawn'] = pad_timers[nearest_pad_id] 
                except KeyError: 
                    logging.warning(f"KeyError during boost pickup processing for player {player_id_idx} at frame index {idx}.")
                    continue
        
        last_time = current_time
    
    return df

def handle_null_values(df: pd.DataFrame) -> pd.DataFrame:
    """Completely eliminate null values with demolition-aware filling"""
    
    # --- Step 1: Prepare 'alive' status columns ---
    # These will be added to the DataFrame later in one go.
    prepared_new_columns = {} 
    for i in range(6):
        alive_col_name = f'p{i}_alive'
        # Check if player's primary position data exists as a proxy for player existence
        if f'p{i}_pos_x' in df.columns:
            # Player is alive if pos_x is not null
            prepared_new_columns[alive_col_name] = (~df[f'p{i}_pos_x'].isnull()).astype(int)
        # If player data (e.g., p{i}_pos_x) doesn't exist, no 'alive' column is created for them here.
        # This means subsequent logic relying on 'p{i}_alive' will only operate if the player was present.

    # --- Step 2: In-place modification of existing columns & preparation of forward vectors ---
    temp_fwd_vector_dfs_to_concat = [] # List to hold DataFrames for each player's forward vectors

    for i in range(6):
        # Skip if player's base data (like pos_x) doesn't exist for this player index 'i'
        if f'p{i}_pos_x' not in df.columns:
            continue
            
        # Determine demo_mask for this player using the 'alive' status prepared earlier.
        # 'alive' status is not yet in df, so access from 'prepared_new_columns'.
        alive_col_name = f'p{i}_alive'
        demo_mask = pd.Series(False, index=df.index) # Default to False (not demoed)
        if alive_col_name in prepared_new_columns: # Check if 'alive' series was prepared
            demo_mask = (prepared_new_columns[alive_col_name] == 0)
        
        # Position handling (modifies df in-place)
        for axis_idx, axis in enumerate(['x', 'y', 'z']):
            pos_col = f'p{i}_pos_{axis}'
            if pos_col in df.columns:
                 df[pos_col] = np.where(
                    demo_mask,
                    DEMO_POSITION[axis_idx],
                    df[pos_col].fillna(DEMO_POSITION[axis_idx]))
            
        # Velocity handling (modifies df in-place)
        for axis_idx, axis in enumerate(['x', 'y', 'z']):
            vel_col = f'p{i}_vel_{axis}'
            if vel_col in df.columns:
                df[vel_col] = np.where(
                    demo_mask,
                    DEMO_VELOCITY[axis_idx],
                    df[vel_col].fillna(0))
        
        # Quaternion handling (modifies df in-place)
        # This ensures quaternion columns are non-null before calculating forward vectors.
        for comp_idx, comp in enumerate(['w','x','y','z']):
            quat_col = f'p{i}_quat_{comp}'
            if quat_col in df.columns:
                df[quat_col] = np.where(
                    demo_mask,
                    DEMO_QUATERNION[comp_idx],
                    df[quat_col].fillna(DEMO_QUATERNION[comp_idx]))
        
        # Boost handling (modifies df in-place)
        boost_col = f'p{i}_boost_amount'
        if boost_col in df.columns:
            df[boost_col] = np.where(
                demo_mask,
                0,
                df[boost_col].fillna(0))
        
        # Forward vector calculation and preparation
        quat_w_col, quat_x_col, quat_y_col, quat_z_col = f'p{i}_quat_w', f'p{i}_quat_x', f'p{i}_quat_y', f'p{i}_quat_z'
        # Quat columns are assumed to be in df and non-null due to the previous step.
        if all(c in df.columns for c in [quat_w_col, quat_x_col, quat_y_col, quat_z_col]):
            q_cols_data = df[[quat_w_col, quat_x_col, quat_y_col, quat_z_col]] # These are now filled
            
            fwd_vectors_series = q_cols_data.apply(
                lambda row: quaternion_to_forward_vector(row[quat_w_col], row[quat_x_col], row[quat_y_col], row[quat_z_col]),
                axis=1
            )
            # Create a DataFrame for this player's forward vectors with final column names
            player_fwd_df = pd.DataFrame(fwd_vectors_series.tolist(), 
                                         index=df.index, 
                                         columns=[f'p{i}_forward_x', f'p{i}_forward_y', f'p{i}_forward_z'])
            temp_fwd_vector_dfs_to_concat.append(player_fwd_df)
        # If quat columns for player 'i' don't exist (should not happen if p{i}_pos_x exists and structure is consistent),
        # their forward vectors won't be computed here. They'll be handled in the post-concat loop.

    # --- Step 3: Add all newly prepared columns to df in fewer operations ---
    # 1. Add 'alive' columns from 'prepared_new_columns'
    if prepared_new_columns:
        df = pd.concat([df, pd.DataFrame(prepared_new_columns, index=df.index)], axis=1)

    # 2. Add all collected forward vector DataFrames
    if temp_fwd_vector_dfs_to_concat:
        all_fwd_vectors_combined_df = pd.concat(temp_fwd_vector_dfs_to_concat, axis=1)
        df = pd.concat([df, all_fwd_vectors_combined_df], axis=1)
    
    # --- Step 4: Post-concatenation fill and demo_mask application for forward vectors ---
    # This loop ensures forward_x/y/z columns exist for all players p0-p5 (if their base data like pos_x existed)
    # and correctly applies demo logic or fills with defaults.
    for i in range(6):
        # Check if player's base data (e.g., pos_x) existed, indicating they were part of processing.
        if f'p{i}_pos_x' not in df.columns:
            continue # No forward vectors needed if player wasn't in the game/data.

        # 'alive' column should now be in df if it was prepared and concatenated.
        alive_col_name = f'p{i}_alive' 
        demo_mask = pd.Series(False, index=df.index) # Default if alive_col not in df
        if alive_col_name in df.columns:
            demo_mask = (df[alive_col_name] == 0)

        for axis_idx, axis in enumerate(['x', 'y', 'z']):
            fwd_col_name = f'p{i}_forward_{axis}'
            # Default forward vector component (1.0 for x, 0.0 for y, z) -> (1,0,0) vector
            # This corresponds to the forward vector from an identity quaternion.
            default_fwd_component = 1.0 if axis == 'x' else 0.0 

            if fwd_col_name in df.columns:
                # Column exists (was concatenated). Fill any potential NaNs and apply demo_mask.
                # NaNs here would be unexpected if quaternion_to_forward_vector is robust.
                df[fwd_col_name] = df[fwd_col_name].fillna(default_fwd_component) 
                df[fwd_col_name] = np.where(
                    demo_mask,
                    default_fwd_component, # Demoed players get (1,0,0)
                    df[fwd_col_name]     # Non-demoed players keep their computed (and now NaN-filled) value
                )
            else:
                # Forward vector column was not created (e.g., quaternions were missing for this player,
                # which is unlikely if p{i}_pos_x exists and quat handling above is complete).
                # Create it now, filled with the default forward vector component.
                df[fwd_col_name] = default_fwd_component
                # If demo_mask is True, this is correct. If False, also correct for missing quats.
                # For explicit clarity, one could re-apply np.where, but it's redundant here.
                # df[fwd_col_name] = np.where(demo_mask, default_fwd_component, default_fwd_component)


    # Handle ball and game columns
    ball_pos_cols = ['ball_pos_x', 'ball_pos_y', 'ball_pos_z']
    if all(col in df.columns for col in ball_pos_cols):
        df.dropna(subset=ball_pos_cols, how='all', inplace=True)
        for col in ball_pos_cols:
            if col in df.columns: # Should exist if outer 'if' passed
                df[col] = df[col].fillna(0)
    
    for pattern, rule in NULL_HANDLING_RULES.items():
        regex_pattern = pattern.replace('[xyz]', '([xyz])').replace('[wxyz]', '([wxyz])').replace('[01]', '([01])')
        for col in df.columns:
            if re.fullmatch(regex_pattern, col):
                if rule['action'] == 'fill':
                    df[col] = df[col].fillna(rule['value'])
                    if 'goal_prev_5s' in col or 'is_overtime' in col:
                        try:
                            df[col] = df[col].astype(bool if isinstance(rule['value'], bool) else int)
                        except ValueError:
                            logging.debug(f"Could not convert column {col} to bool/int after fillna, leaving as {df[col].dtype}")
                    elif 'ball_hit_team_num' in col or 'seconds_remaining' in col:
                        try:
                            df[col] = pd.to_numeric(df[col], errors='raise')
                        except ValueError:
                            logging.debug(f"Could not convert column {col} to numeric after fillna, leaving as {df[col].dtype}")
                    elif df[col].dtype == 'object':
                         df[col] = df[col].infer_objects(copy=False)
                # 'drop' action for NULL_HANDLING_RULES is not explicitly handled here, assuming pre-dropna or different logic.
    
    if df.isnull().any().any():
        remaining_null_cols = df.columns[df.isnull().any()].tolist()
        logging.debug(f"Nulls still present in: {remaining_null_cols} after specific handling. Applying general fillna(0).")
        df.fillna(0, inplace=True) 
    
    return df

def filter_to_active_play(df: pd.DataFrame, analyzer_data: dict) -> pd.DataFrame:
    """
    Precisely filters the DataFrame to active gameplay periods using the authoritative
    'start_frame' and 'goal_frame'/'end_frame' from carball's analyzer.json.
    This handles all regulation play and post-goal dead time perfectly.
    """
    if df.empty:
        return df

    logging.info("Filtering to active play periods using analyzer.json 'start_frame' and 'goal_frame'/'end_frame'...")
    
    gameplay_periods = analyzer_data.get('gameplay_periods')
    if not gameplay_periods:
        logging.warning("No 'gameplay_periods' found in analyzer.json. Cannot filter for active play.")
        return pd.DataFrame(columns=df.columns)

    list_of_active_dfs = []
    
    for period in gameplay_periods:
        start_frame = period.get('start_frame')
        end_of_play_frame = period.get('goal_frame') or period.get('end_frame')

        if start_frame is None or end_of_play_frame is None:
            logging.warning(f"Skipping invalid gameplay period (missing start/end frame): {period}")
            continue
            
        period_mask = (df['original_frame'] >= start_frame) & (df['original_frame'] <= end_of_play_frame)
        active_segment = df.loc[period_mask]
        
        if not active_segment.empty:
            list_of_active_dfs.append(active_segment)
    
    if not list_of_active_dfs:
        logging.warning("No valid active play segments were extracted from analyzer.json.")
        return pd.DataFrame(columns=df.columns)

    final_df = pd.concat(list_of_active_dfs, ignore_index=True)
    logging.info(f"Kept {len(final_df)} frames from {len(list_of_active_dfs)} primary active play period(s).")
    
    return final_df

def trim_overtime_kickoff_countdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Finds the start of an overtime period and removes the static countdown frames
    until any player's x or y position changes. This is the most robust method.

    Args:
        df: DataFrame that has been cleaned by filter_to_active_play.

    Returns:
        A DataFrame with the overtime kickoff countdown frames removed.
    """
    if df.empty or 'is_overtime' not in df.columns or not df['is_overtime'].any():
        # No overtime in the dataset, so no action is needed.
        return df

    # Find the very first frame where overtime begins IN THE CURRENT DATAFRAME.
    # Using .diff() is robust to find the exact switch point.
    overtime_start_mask = df['is_overtime'].diff().fillna(False)
    
    if not overtime_start_mask.any():
        # This can happen if the entire DF is overtime after the first filter.
        # We assume the first frame is the start of the OT kickoff sequence.
        if df['is_overtime'].iloc[0]:
            ot_start_idx = df.index[0]
        else: # Should not happen, but as a safeguard.
            return df
    else:
        ot_start_idx = overtime_start_mask.idxmax()

    logging.info(f"Overtime start detected at index {ot_start_idx}. "
                 f"Searching for first player position change to trim countdown.")

    # Define the columns to check for any movement.
    pos_cols = [f'p{i}_pos_{axis}' for i in range(6) for axis in ['x', 'y']]
    
    # Ensure all required position columns exist to avoid KeyErrors.
    pos_cols_exist = all(col in df.columns for col in pos_cols)
    if not pos_cols_exist:
        logging.error("Cannot trim OT countdown: missing required player position columns.")
        return df

    # Get the starting positions of all players at the beginning of the OT countdown.
    try:
        start_positions = df.loc[ot_start_idx, pos_cols]
    except KeyError:
        logging.error(f"Could not get starting positions at index {ot_start_idx}.")
        return df
    
    # Define the search segment starting from the OT start.
    search_segment = df.loc[ot_start_idx:]
    
    # Check for any change in any player's X or Y position.
    # The .ne() method compares the DataFrame segment with the Series of start positions.
    # It broadcasts the Series across the rows for an efficient comparison.
    pos_changed_df = search_segment[pos_cols].ne(start_positions)
    any_player_moved_mask = pos_changed_df.any(axis=1)
    
    # The "Go!" frame is the first frame where any player has moved.
    if any_player_moved_mask.any():
        go_frame_idx = any_player_moved_mask.idxmax()
        
        # We need to keep the data before the OT countdown, plus the data from the "go" frame onwards.
        # Slicing with .loc is safest here as the index might not be contiguous.
        df_before_ot_countdown = df.loc[df.index < ot_start_idx]
        df_after_countdown = df.loc[df.index >= go_frame_idx]
        
        num_removed = len(df) - (len(df_before_ot_countdown) + len(df_after_countdown))
        logging.info(f"Removing {num_removed} frames from overtime kickoff countdown.")
        
        return pd.concat([df_before_ot_countdown, df_after_countdown], ignore_index=True)

    # If no movement is found (highly unlikely edge case), we don't remove anything to be safe.
    logging.warning("No player movement detected after start of overtime. OT countdown not trimmed.")
    return df

def adjust_seconds_remaining_for_overtime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjusts 'seconds_remaining' for overtime and ensures the final column is an integer type.
    In overtime, the 'seconds_remaining' value (which counts up from 0) is made negative.

    Args:
        df: The DataFrame to be processed. Must contain 'is_overtime' and 'seconds_remaining'.

    Returns:
        The DataFrame with the adjusted 'seconds_remaining' column.
    """
    required_cols = {'is_overtime', 'seconds_remaining'}
    if not required_cols.issubset(df.columns):
        logging.warning("adjust_seconds_remaining_for_overtime: Missing required columns. Skipping.")
        return df

    # Create a boolean mask for all rows that are in overtime.
    # astype(bool) handles cases where it might be 0/1 or True/False.
    overtime_mask = df['is_overtime'].astype(bool)

    # If there's no overtime, just ensure the column is int and return.
    if not overtime_mask.any():
        df['seconds_remaining'] = df['seconds_remaining'].round().astype('int64')
        return df

    logging.info("Overtime detected. Adjusting 'seconds_remaining' to be negative for OT periods.")
    
    # Use np.where for a conditional assignment based on your logic:
    # Condition: Is it overtime?
    # If True:  Take the 'seconds_remaining' value and make it negative.
    # If False: Keep the original 'seconds_remaining' value.
    df['seconds_remaining'] = np.where(
        overtime_mask,
        -df['seconds_remaining'],
        df['seconds_remaining']
    )

    # Finally, round and cast the entire column to int64 to ensure a consistent integer type.
    # This handles both the newly negative OT values and the existing positive regulation values.
    df['seconds_remaining'] = df['seconds_remaining'].round().astype('int64')

    return df
    
def find_replay_files(root_dir: Path) -> List[Path]:
    """
    Recursively find all .replay files in directory tree.
    
    Args:
        root_dir: Directory to search for replay files
        
    Returns:
        List of Path objects to .replay files
    """
    try:
        return list(root_dir.rglob("*.replay"))
    except Exception as e:
        logging.error(f"Error finding replay files: {str(e)}")
        return []

def run_carball(replay_file: Path, output_dir: Path) -> bool:
    """
    Execute carball.exe to parse replay file into structured data.
    
    Args:
        replay_file: Path to .replay file
        output_dir: Directory to save carball output
        
    Returns:
        True if successful, False otherwise
    """
    command = [
        str(CARBALL_EXE), 
        "parquet", 
        "-i", str(replay_file), 
        "-o", str(output_dir)
    ]
    
    try:
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=True,
            creationflags=subprocess.CREATE_NO_WINDOW # For Windows, to hide console
        )
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"carball.exe failed for {replay_file.name}:\n{e.stderr}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error running carball for {replay_file.name}: {str(e)}")
        return False

def process_player_data(output_dir: Path, 
                       metadata: dict) -> Tuple[List[pd.DataFrame], List[str]]:
    valid_players = []
    skipped_observers_info = [] 

    # Loop to filter valid_players and collect skipped_observers_info
    for player_meta in metadata.get('players', []):
        # ... (as corrected before) ...
        player_name = player_meta.get('name', 'Unknown')
        player_id = player_meta.get('unique_id', 'N/A')
        is_orange_val = player_meta.get('is_orange')
        is_orange_type = type(is_orange_val)

        if not all(k in player_meta for k in ['is_orange', 'unique_id', 'name']) or \
           player_id == 'N/A':
            logging.warning(
                f"Player {player_name} (ID: {player_id}) has fundamentally incomplete metadata. Skipping."
            )
            continue
        
        if not isinstance(is_orange_val, bool):
            skipped_observers_info.append(
                f"{player_name} (ID: {player_id}, is_orange_val: '{is_orange_val}', type: {is_orange_type.__name__})"
            )
            continue 
            
        valid_players.append(player_meta)
    
    if skipped_observers_info:
        logging.info(
            f"For replay, {len(skipped_observers_info)} players/entries were skipped due to invalid 'is_orange' status (likely observers): "
            f"[{'; '.join(skipped_observers_info)}]."
        )

    players_sorted = sorted(valid_players, key=lambda p: (p['is_orange'], str(p['unique_id'])))
    
    blue_players = [p for p in players_sorted if not p['is_orange']]
    orange_players = [p for p in players_sorted if p['is_orange']]

    if not (len(blue_players) == 3 and len(orange_players) == 3):
        player_names_for_log = [p.get('name', 'Unknown') for p in players_sorted] # Use .get for safety
        logging.warning(
            f"Replay does not have exactly 3 blue and 3 orange active players. "
            f"Found {len(blue_players)} blue, {len(orange_players)} orange. "
            f"Active players considered: {player_names_for_log}. Skipping this entire replay."
        )
        return [], [] 

    # --- INITIALIZE all_player_columns_set and player_dfs_map HERE ---
    player_dfs_map = {} 
    all_player_columns_set = set() # <<< THIS IS THE FIX: Initialize as an empty set

    # Loop to process individual player parquet files
    for assigned_idx, player_meta in enumerate(players_sorted): # players_sorted now guaranteed to be 6 players
        player_carball_id = str(player_meta['unique_id'])
        player_name = player_meta.get('name', 'Unknown') # Get name for logging within this loop too
        player_file = output_dir / f"player_{player_carball_id}.parquet"
        
        if not player_file.exists():
            logging.warning(f"Parquet file not found for ACTIVE player {player_name} (ID: {player_carball_id}). This is unexpected. Skipping player processing for this replay.")
            # If an active player's file is missing, we must skip the whole replay to maintain 3v3
            return [], [] # Skip the entire replay
            
        try:
            player_df_raw = pd.read_parquet(player_file)
            cols_to_keep_rename = {
                'pos_x': f'p{assigned_idx}_pos_x', 'pos_y': f'p{assigned_idx}_pos_y', 'pos_z': f'p{assigned_idx}_pos_z',
                'vel_x': f'p{assigned_idx}_vel_x', 'vel_y': f'p{assigned_idx}_vel_y', 'vel_z': f'p{assigned_idx}_vel_z',
                'quat_w': f'p{assigned_idx}_quat_w', 'quat_x': f'p{assigned_idx}_quat_x',
                'quat_y': f'p{assigned_idx}_quat_y', 'quat_z': f'p{assigned_idx}_quat_z',
                'boost_amount': f'p{assigned_idx}_boost_amount',
                'boost_pickup': f'p{assigned_idx}_boost_pickup'
            }
            existing_cols_to_rename = {orig_col: new_name 
                                       for orig_col, new_name in cols_to_keep_rename.items() 
                                       if orig_col in player_df_raw.columns}
            
            if len(existing_cols_to_rename) < len(cols_to_keep_rename):
                 missing_orig_cols = set(cols_to_keep_rename.keys()) - set(existing_cols_to_rename.keys())
                 logging.warning(f"Player {player_name} (ID: {player_carball_id}) is missing expected columns in parquet: {missing_orig_cols}. Will proceed with available data.")

            player_df_processed = player_df_raw[list(existing_cols_to_rename.keys())].rename(columns=existing_cols_to_rename)
            player_df_processed[f'p{assigned_idx}_team'] = 1 if player_meta['is_orange'] else 0
            
            # Ensure quaternion columns exist, filling with defaults if missing
            for q_comp_idx, q_comp_val in enumerate(['w','x','y','z']):
                q_col_name = f'p{assigned_idx}_quat_{q_comp_val}'
                if q_col_name not in player_df_processed.columns:
                     player_df_processed[q_col_name] = DEMO_QUATERNION[q_comp_idx] # Fill with component of identity quaternion

            player_dfs_map[f'p{assigned_idx}'] = player_df_processed
            all_player_columns_set.update(player_df_processed.columns.tolist()) # Now this is safe
        except Exception as e:
            # This error is critical for the player, might compromise 3v3.
            logging.error(f"Error processing parquet for player {player_name} (ID: {player_carball_id}): {e}. Skipping this replay.")
            return [], [] # Skip the entire replay

    final_ordered_dfs = [player_dfs_map[f'p{i}'] for i in range(6) if f'p{i}' in player_dfs_map]
    
    if len(final_ordered_dfs) != 6:
        logging.warning(f"After attempting to load player parquet files, ended up with {len(final_ordered_dfs)} players instead of 6 "
                        f"for replay. This might be due to earlier skips or missing .parquet files. Skipping replay.")
        return [], []

    return final_ordered_dfs, sorted(list(all_player_columns_set)) # Now this is safe

def process_replay(replay_file: Path, individual_csv_output_path: Path, replay_id: int) -> Optional[pd.DataFrame]:
    """
    Full processing pipeline for a single replay file, using a two-step cleaning
    process to correctly handle all game phases.
    """
    replay_start_time_mono = time.monotonic()
    output_folder_name = CARBALL_OUTPUT_FOLDER_FORMAT.format(stem=replay_file.stem)
    output_dir = replay_file.parent / output_folder_name
    
    # ... (code to create/clean output_dir and run_carball) ...
    if output_dir.exists():
        try:
            import shutil
            shutil.rmtree(output_dir)
        except Exception as e:
            logging.warning(f"Could not remove old output directory {output_dir}: {e}.")
    output_dir.mkdir(exist_ok=True)
    logging.info(f"Processing {replay_file.name} (ID: {replay_id})...")
    if not run_carball(replay_file, output_dir):
        return None

    try:
        # --- 1. Load All Raw Data Sources ---
        # ... (Identical to previous response: load metadata, players, game, ball) ...
        metadata_path = output_dir / 'metadata.json'
        if not metadata_path.exists(): return None
        with open(metadata_path, 'r', encoding='utf8') as f: metadata = json.load(f)
        ordered_player_dfs, _ = process_player_data(output_dir, metadata)
        if not ordered_player_dfs: return None
        game_parquet_path = output_dir / '__game.parquet'
        if not game_parquet_path.exists(): return None
        game_df_raw = pd.read_parquet(game_parquet_path)
        gameplay_tick_rate = get_empirical_tick_rate(game_df_raw, expected_rates=[30, 60, 120])
        game_df = game_df_raw.drop(columns=['delta', 'replicated_game_state_time_remaining', 'ball_has_been_hit'], errors='ignore')
        ball_parquet_path = output_dir / '__ball.parquet'
        if not ball_parquet_path.exists(): return None
        ball_df = pd.read_parquet(ball_parquet_path).add_prefix('ball_').drop(
            columns=['ball_quat_w', 'ball_quat_x', 'ball_quat_y', 'ball_quat_z', 'ball_ang_vel_x', 'ball_ang_vel_y', 'ball_ang_vel_z', 'ball_is_sleeping', 'ball_has_been_hit'], errors='ignore')

        # --- 2. Combine and Pre-process Full Dataset ---
        # ... (Identical to previous response: concat, handle_nulls, adjust_seconds, add context) ...
        dfs_to_concat = [game_df] + ordered_player_dfs + [ball_df]
        combined_df = pd.concat(dfs_to_concat, axis=1)
        combined_df['original_frame'] = np.arange(len(combined_df))
        combined_df = handle_null_values(combined_df)
        combined_df = adjust_seconds_remaining_for_overtime(combined_df)
        goal_events = []
        raw_goals = metadata.get('game', {}).get('goals', [])
        for g in raw_goals:
            if g.get('frame') is not None and g.get('is_orange') is not None:
                goal_time = g.get('time') or (combined_df.iloc[g['frame']]['time'] if 'time' in combined_df.columns and 0 <= g['frame'] < len(combined_df) else None)
                if goal_time is not None:
                    goal_events.append(GoalEvent(frame=g['frame'], time=float(goal_time), team=1 if g['is_orange'] else 0))
        combined_df = add_score_context_columns(combined_df, goal_events)
        combined_df = add_replay_id_column(combined_df, f"replay_{replay_id:05d}")
        combined_df = update_boost_pad_timers(combined_df)

        # --- 3. DEFINITIVE TWO-STEP GAMEPLAY CLEANING ---
        # Step 3a: Main filtering using analyzer.json for regulation play
        analyzer_path = output_dir / "analyzer.json"
        if not analyzer_path.exists():
            logging.error(f"analyzer.json not found for {replay_file.name}.")
            return None
        with open(analyzer_path, "r", encoding="utf8") as f:
            analyzer_data = json.load(f)

        active_play_df = filter_to_active_play(combined_df, analyzer_data)

        if active_play_df.empty:
            logging.warning(f"No active gameplay frames found after main filtering for {replay_file.name}.")
            return None
            
        # Step 3b: Specialized filter for the OT kickoff countdown
        active_play_df = trim_overtime_kickoff_countdown(active_play_df)

        if active_play_df.empty:
            logging.warning(f"No active gameplay frames remaining after OT kickoff removal for {replay_file.name}.")
            return None

        # --- 4. Final Processing on Cleaned Data ---
        # ... (The rest of the function is identical to the previous response) ...
        goal_label_cols = [f'team_{t}_goal_in_event_window' for t in [0, 1]]
        active_play_df[goal_label_cols] = 0
        for goal in goal_events:
            goal_time_float = goal.time
            mask = (active_play_df['time'] >= goal_time_float - GOAL_ANTICIPATION_WINDOW_SECONDS) & \
                   (active_play_df['time'] <= goal_time_float)
            active_play_df.loc[mask, f'team_{goal.team}_goal_in_event_window'] = 1
        
        final_df = downsample_data(active_play_df, original_hz=gameplay_tick_rate, event_columns=goal_label_cols)
        
        if final_df.empty: return None
        
        pos_count = sum(final_df[col].sum() for col in goal_label_cols if col in final_df.columns)
        neg_count = len(final_df) - pos_count
        with counts_lock:
            total_counts["positive"] += int(pos_count)
            total_counts["negative"] += int(neg_count)
            
        new_dist_cols_data = {}
        ball_pos_cols = ['ball_pos_x', 'ball_pos_y', 'ball_pos_z']
        if all(col in final_df.columns for col in ball_pos_cols):
            ball_positions = final_df[ball_pos_cols].values
            for i in range(6):
                player_pos_cols = [f'p{i}_pos_x', f'p{i}_pos_y', f'p{i}_pos_z']
                if all(col in final_df.columns for col in player_pos_cols):
                    player_positions = final_df[player_pos_cols].values
                    distances = np.linalg.norm(player_positions - ball_positions, axis=1)
                    new_dist_cols_data[f'p{i}_dist_to_ball'] = np.round(distances, 2)
        final_df = final_df.assign(**new_dist_cols_data)
        
        cols_to_drop = ["original_frame", "frame", "is_overtime", 
                        *[f'p{i}_quat_{c}' for i in range(6) for c in ['w', 'x', 'y', 'z']], 
                        *[f'p{i}_boost_pickup' for i in range(6)]]
        final_df.drop(columns=[col for col in cols_to_drop if col in final_df.columns], inplace=True)
        
        context_cols = ['replay_id', 'blue_score', 'orange_score', 'score_difference', 'seconds_remaining']
        ordered_context_cols = [col for col in context_cols if col in final_df.columns]
        other_cols = sorted([col for col in final_df.columns if col not in ordered_context_cols])
        final_df = final_df[ordered_context_cols + other_cols]
        
        individual_csv_filename_only = f"{final_df['replay_id'].iloc[0]}.csv"
        csv_path = individual_csv_output_path / individual_csv_filename_only
        final_df.to_csv(csv_path, index=False, float_format='%.4f')
        
        processing_time = time.monotonic() - replay_start_time_mono
        logging.info(f"Successfully processed {replay_file.name} in {processing_time:.1f}s. Saved to {csv_path.name}")
        
        return final_df

    except Exception as e:
        logging.exception(f"Critical failure processing {replay_file.name}: {str(e)}")
        return None
    finally:
        if output_dir.exists():
            try:
                import shutil
                shutil.rmtree(output_dir)
            except Exception as e_clean:
                logging.warning(f"Failed to cleanup {output_dir} after processing: {e_clean}")

# ==================================================================
# Main Execution
# ==================================================================
def main():
    """Main execution function"""
    configure_logging() # Setup logging first
    
    global COMBINED_DATASET_CSV_FILENAME, PROCESSING_SUMMARY_FILENAME

    # Validate carball executable exists
    if not CARBALL_EXE.exists():
        logging.critical(f"carball.exe not found at {CARBALL_EXE}. Please check the path.")
        sys.exit(1)

    # Construct dynamic filenames
    global INDIVIDUAL_REPLAY_CSV_FILENAME_FORMAT, COMBINED_DATASET_CSV_FILENAME, PROCESSING_SUMMARY_FILENAME
    
    if POSITIVE_STATE_TARGET_HZ == NEGATIVE_STATE_TARGET_HZ:
        freq_str = f"{POSITIVE_STATE_TARGET_HZ}hz"
    else:
        freq_str = f"P{POSITIVE_STATE_TARGET_HZ}N{NEGATIVE_STATE_TARGET_HZ}hz"
    
    time_window_str = f"{GOAL_ANTICIPATION_WINDOW_SECONDS}sec"
    replay_group_name_raw = PARENT_DIR.name 
    replay_group_name_sanitized = re.sub(r'[^\w\-.]+', '', replay_group_name_raw.replace(' ', '_'))
    if not replay_group_name_sanitized:
        replay_group_name_sanitized = "replays" 

    COMBINED_DATASET_CSV_FILENAME = f"dataset_{freq_str}_{time_window_str}_{replay_group_name_sanitized}.csv"
    PROCESSING_SUMMARY_FILENAME = f"processing_summary_{freq_str}_{time_window_str}_{replay_group_name_sanitized}.txt"
    
    logging.info(f"Individual CSVs will be named like: replay_00000.csv, replay_00001.csv, etc.")
    logging.info(f"Output combined dataset will be: {COMBINED_DATASET_CSV_FILENAME}")
    logging.info(f"Output summary file will be: {PROCESSING_SUMMARY_FILENAME}")

    individual_csvs_output_dir = PARENT_DIR / INDIVIDUAL_CSVS_SUBFOLDER_NAME
    try:
        individual_csvs_output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Individual processed replay CSVs will be saved in: {individual_csvs_output_dir}")
    except OSError as e:
        logging.critical(f"Could not create directory for individual CSVs: {individual_csvs_output_dir}. Error: {e}")
        sys.exit(1)

    overall_start_time = time.monotonic()
    replays = find_replay_files(PARENT_DIR)

    if not replays:
        logging.warning(f"No .replay files found in {PARENT_DIR} or its subdirectories.")
        return

    logging.info(f"Found {len(replays)} replay files. Processing with PositiveStateTargetHZ={POSITIVE_STATE_TARGET_HZ}, NegativeStateTargetHZ={NEGATIVE_STATE_TARGET_HZ}.")

    all_processed_dfs = []
    successful_replays_stats = []
    failed_replay_names = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_replay = {}
        for replay_file in replays:
            # Generate the next unique, thread-safe integer ID
            with replay_id_lock:
                current_replay_id = next(replay_id_counter)
            
            logging.debug(f"Assigning replay_id {current_replay_id} to {replay_file.name}")
            
            # Submit the job with the new integer ID
            future = executor.submit(
                process_replay,
                replay_file,
                individual_csvs_output_dir,
                current_replay_id  # Pass the integer ID
            )
            future_to_replay[future] = replay_file
        
        for i, future in enumerate(as_completed(future_to_replay)):
            replay_file_path = future_to_replay[future]
            replay_name = replay_file_path.name
            current_num = i + 1
            logging.info(f"--- [{current_num}/{len(replays)}] Checking result for: {replay_name} ---")
            try:
                # Retrieve the per-replay processing time if process_replay returned it.
                # For now, we'll need to get it from the future or pass it back from process_replay.
                # The current process_replay calculates it but doesn't return it with the DataFrame.
                # For simplicity, we'll keep ReplayStats.processing_time as 0 for now.
                
                result_df = future.result() 
                if result_df is not None and not result_df.empty:
                    all_processed_dfs.append(result_df)
                    successful_replays_stats.append(ReplayStats(
                        replay_name=replay_name,
                        rows=len(result_df),
                        processing_time=0, # Placeholder, actual time captured inside process_replay log
                        success=True
                    ))
                elif result_df is None: 
                    failed_replay_names.append(replay_name)
                    logging.warning(f"{replay_name} processing returned None.")
                else: 
                    failed_replay_names.append(replay_name)
                    logging.warning(f"{replay_name} processing resulted in an empty DataFrame.")

            except Exception as e: 
                failed_replay_names.append(replay_name)
                logging.error(f"Exception during processing of {replay_name}: {type(e).__name__} - {str(e)}")


    if all_processed_dfs:
        logging.info(f"Concatenating {len(all_processed_dfs)} processed replay DataFrames...")
        final_combined_df = pd.concat(all_processed_dfs, ignore_index=True)
        output_csv_path = PARENT_DIR / COMBINED_DATASET_CSV_FILENAME
        final_combined_df.to_csv(output_csv_path, index=False)
        logging.info(f"Successfully saved combined dataset to: {output_csv_path} ({len(final_combined_df)} rows)")
    else:
        logging.warning("No replay data was successfully processed to form a combined dataset.")

    generate_summary_report(
        replays_found_paths=replays, 
        successful_stats_list=successful_replays_stats,
        failed_replay_names_list=failed_replay_names,
        processing_start_time=overall_start_time,
        summary_output_dir=PARENT_DIR # Save summary in the root of replay processing
    )

def generate_summary_report(replays_found_paths: List[Path],
                          successful_stats_list: List[ReplayStats],
                          failed_replay_names_list: List[str],
                          processing_start_time: float,
                          summary_output_dir: Path):
    """
    Generate a comprehensive processing summary report.
    """
    summary_lines = []
    total_processing_duration_seconds = time.monotonic() - processing_start_time
    minutes, seconds = divmod(total_processing_duration_seconds, 60)

    summary_lines.append("========================================")
    summary_lines.append(" Rocket League Replay Processing Summary")
    summary_lines.append("========================================")
    summary_lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"Total replays found: {len(replays_found_paths)}")
    summary_lines.append(f"Successfully processed: {len(successful_stats_list)}")
    summary_lines.append(f"Failed or empty replays: {len(failed_replay_names_list)}")
    summary_lines.append(f"Total processing time: {int(minutes)}m {seconds:.1f}s")
    summary_lines.append(f"Configuration: POSITIVE_STATE_TARGET_HZ={POSITIVE_STATE_TARGET_HZ}, NEGATIVE_STATE_TARGET_HZ={NEGATIVE_STATE_TARGET_HZ}, GOAL_ANTICIPATION_WINDOW_SECONDS={GOAL_ANTICIPATION_WINDOW_SECONDS}")


    if failed_replay_names_list:
        summary_lines.append("\n--- Failed/Empty Replays ---")
        for name in failed_replay_names_list:
            summary_lines.append(f" - {name}")

    if successful_stats_list:
        summary_lines.append("\n--- Successfully Processed Replays ---")
        summary_lines.append(f"{'Replay Name':<50}{'Rows':>10}")
        total_rows_processed = 0
        for stat in successful_stats_list:
            summary_lines.append(
                f"{stat.replay_name:<50}{stat.rows:>10}" 
            )
            total_rows_processed += stat.rows
        summary_lines.append(f"\nTotal rows in combined dataset from successful replays: {total_rows_processed}")


    with counts_lock:
        pos_total = total_counts["positive"]
        neg_total = total_counts["negative"]
    
    total_samples = pos_total + neg_total
    imbalance_ratio_overall = neg_total / max(1, pos_total) if pos_total > 0 else float('inf')

    summary_lines.append("\n--- Aggregate Class Balance (from all successfully processed frames) ---")
    summary_lines.append(f"Positive samples (goal imminent): {pos_total}")
    summary_lines.append(f"Negative samples (other gameplay): {neg_total}")
    summary_lines.append(f"Total labeled samples: {total_samples}")
    summary_lines.append(f"Overall class imbalance ratio (Negative/Positive): {imbalance_ratio_overall:.1f}:1")

    summary_file_path = summary_output_dir / PROCESSING_SUMMARY_FILENAME
    try:
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(summary_lines))
        logging.info(f"Processing summary report written to: {summary_file_path}")
    except IOError as e:
        logging.error(f"Failed to write summary report to {summary_file_path}: {e}")


if __name__ == "__main__":
    main()
    