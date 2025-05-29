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

pd.set_option('future.no_silent_downcasting', True)

# ==================================================================
# Global Configuration
# ==================================================================
# Using global variables for configuration that would typically come from a config file
SCRIPT_DIR = Path(__file__).resolve().parent
CARBALL_EXE = SCRIPT_DIR / "carball.exe"  # Path to carball executable
PARENT_DIR = Path(r"C:\\Users\\serda\\Desktop\\MS-Thesis-Oguz-Arslan\\converting-replay-files\\example-resources\\replay-files")  # Root directory containing replays
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
    'team_[01]_goal_prev_5s': {'action': 'fill', 'value': 0}
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


def clean_post_goal_frames_using_ball_pos(df: pd.DataFrame,
                                         goal_event_original_frames: List[int]) -> pd.DataFrame:
    """
    Remove frames between goals and subsequent kickoffs (ball at 0,0).
    Keeps the goal frame and the first frame where ball is at (x=0, y=0).
    
    Args:
        df: DataFrame. Must have 'original_frame', 'ball_pos_x', 'ball_pos_y'.
        goal_event_original_frames: List of ORIGINAL_FRAME numbers where goals occurred.
        
    Returns:
        Filtered DataFrame.
    """
    required_cols = {'original_frame', 'ball_pos_x', 'ball_pos_y'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        logging.warning(f"clean_post_goal_frames_using_ball_pos: Missing required columns: {missing}. Skipping.")
        return df
    if df.empty:
        return df

    keep_mask = pd.Series(True, index=df.index)
    sorted_goal_original_frames = sorted(goal_event_original_frames)

    for goal_orig_frame_num in sorted_goal_original_frames:
        logging.debug(f"Processing goal (ball_pos method) with original_frame: {goal_orig_frame_num}")

        goal_frame_rows = df[df['original_frame'] == goal_orig_frame_num]
        if goal_frame_rows.empty:
            logging.debug(f"Goal original_frame {goal_orig_frame_num} not found. Skipping.")
            continue
        
        goal_df_label = goal_frame_rows.index[0]
        try:
            goal_df_pos = df.index.get_loc(goal_df_label)
        except KeyError:
            logging.error(f"Error getting position for goal_df_label {goal_df_label}.")
            continue
        
        logging.debug(f"Goal found: original_frame={goal_orig_frame_num}, df_label={goal_df_label}, df_pos={goal_df_pos}, "
                      f"ball_pos=({df.loc[goal_df_label, 'ball_pos_x']:.0f}, {df.loc[goal_df_label, 'ball_pos_y']:.0f})")

        search_kickoff_start_pos = goal_df_pos + 1
        if search_kickoff_start_pos >= len(df.index):
            logging.debug(f"Goal {goal_orig_frame_num} near end of DF. No frames after to search.")
            continue
            
        df_segment_to_search_kickoff = df.iloc[search_kickoff_start_pos:]
        if df_segment_to_search_kickoff.empty:
            logging.debug(f"No frames after goal {goal_orig_frame_num} to search for kickoff.")
            continue

        # KICKOFF CONDITION: ball_pos_x is 0 AND ball_pos_y is 0
        # Using a small tolerance for floating point comparison might be safer,
        # but carball often gives exact 0s. Let's start with exact.
        kickoff_condition_met_series = (df_segment_to_search_kickoff['ball_pos_x'] == 0) & \
                                       (df_segment_to_search_kickoff['ball_pos_y'] == 0)
        
        if not kickoff_condition_met_series.any():
            logging.debug(f"No kickoff (ball_pos_x/y == 0) found after goal {goal_orig_frame_num}.")
            continue
            
        kickoff_df_label = kickoff_condition_met_series.idxmax()
        # Sanity check (already done by condition, but good for belt-and-suspenders if using tolerance)
        # if not ((df.loc[kickoff_df_label, 'ball_pos_x'] == 0) and (df.loc[kickoff_df_label, 'ball_pos_y'] == 0)):
        #     logging.warning(...)
        #     continue
        
        try:
            kickoff_df_pos = df.index.get_loc(kickoff_df_label)
        except KeyError:
            logging.error(f"Error getting position for kickoff_df_label {kickoff_df_label}.")
            continue

        logging.debug(f"Kickoff (ball_pos_x/y=0) found for goal {goal_orig_frame_num}: "
                      f"df_label={kickoff_df_label}, df_pos={kickoff_df_pos}, "
                      f"ball_pos=({df.loc[kickoff_df_label, 'ball_pos_x']:.0f}, {df.loc[kickoff_df_label, 'ball_pos_y']:.0f}), "
                      f"orig_frame={df.loc[kickoff_df_label, 'original_frame']}")

        remove_slice_start_pos = goal_df_pos + 1
        remove_slice_end_exclusive_pos = kickoff_df_pos 

        if remove_slice_start_pos < remove_slice_end_exclusive_pos:
            indices_to_drop_labels = df.iloc[remove_slice_start_pos : remove_slice_end_exclusive_pos].index
            keep_mask.loc[indices_to_drop_labels] = False
            logging.info(f"Marked {len(indices_to_drop_labels)} frames for removal (ball_pos method) between "
                         f"goal (orig {goal_orig_frame_num}, pos {goal_df_pos}) and "
                         f"kickoff (orig {df.loc[kickoff_df_label, 'original_frame']}, pos {kickoff_df_pos}). "
                         f"Pos range removed: {remove_slice_start_pos} to {remove_slice_end_exclusive_pos - 1}.")
        else:
            logging.debug(f"No frames to remove (ball_pos method) between goal (pos {goal_df_pos}) and kickoff (pos {kickoff_df_pos}). Adjacent or overlap.")

    num_removed = (~keep_mask).sum()
    logging.info(f"Total frames marked for removal by clean_post_goal_frames_using_ball_pos: {num_removed} out of {len(df)}.")
    return df[keep_mask].copy()

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


def process_replay(replay_file: Path, individual_csv_output_path: Path) -> Optional[pd.DataFrame]:
    """
    Full processing pipeline for a single replay file.
    
    Args:
        replay_file: Path to the replay file to process
        individual_csv_output_path: Path to the directory where individual CSVs should be saved.
        
    Returns:
        Processed DataFrame if successful, None otherwise
    """
    replay_start_time_mono = time.monotonic() # For precise duration
    output_folder_name = CARBALL_OUTPUT_FOLDER_FORMAT.format(stem=replay_file.stem)
    # Place carball's temporary output inside the replay's original directory
    output_dir = replay_file.parent / output_folder_name 
    
    if output_dir.exists():
        try:
            import shutil
            shutil.rmtree(output_dir)
        except Exception as e:
            logging.warning(f"Could not remove old output directory {output_dir}: {e}. Files might be in use.")
    output_dir.mkdir(exist_ok=True)
    
    logging.info(f"Processing {replay_file.name}...")
    
    if not run_carball(replay_file, output_dir):
        return None

    try:
        metadata_path = output_dir / 'metadata.json'
        if not metadata_path.exists():
            logging.error(f"metadata.json not found in {output_dir} for {replay_file.name}")
            return None
        with open(metadata_path, 'r', encoding='utf8') as f:
            metadata = json.load(f)
        
        ordered_player_dfs, _ = process_player_data(output_dir, metadata)
        if not ordered_player_dfs:
            logging.warning(f"Skipping replay {replay_file.name} due to player data processing issues (e.g., not 3v3).")
            if output_dir.exists():
                try:
                    import shutil; shutil.rmtree(output_dir)
                except Exception as e_clean:
                    logging.warning(f"Failed to cleanup {output_dir} after player data issue: {e_clean}")
            return None
        
        game_parquet_path = output_dir / '__game.parquet'
        if not game_parquet_path.exists():
            logging.error(f"__game.parquet not found for {replay_file.name}")
            return None
        game_df = pd.read_parquet(game_parquet_path).drop(
            columns=['delta', 'replicated_game_state_time_remaining', 'ball_has_been_hit'],
            errors='ignore'
        )
        
        ball_parquet_path = output_dir / '__ball.parquet'
        if not ball_parquet_path.exists():
            logging.error(f"__ball.parquet not found for {replay_file.name}")
            return None
        ball_df = pd.read_parquet(ball_parquet_path).add_prefix('ball_').drop(
            columns=['ball_quat_w', 'ball_quat_x', 'ball_quat_y', 'ball_quat_z',
                     'ball_ang_vel_x', 'ball_ang_vel_y', 'ball_ang_vel_z', 
                     'ball_is_sleeping', 'ball_has_been_hit'],
            errors='ignore'
        )

        dfs_to_concat = [game_df] + ordered_player_dfs + [ball_df]
        try:
            combined_df = pd.concat(dfs_to_concat, axis=1).round(2)
        except Exception as e:
            logging.error(f"Error concatenating DataFrames for {replay_file.name}: {e}")
            # Log lengths for debugging
            logging.error(f"Lengths: GameDF={len(game_df)}, BallDF={len(ball_df)}, PlayerDFs={[len(pdf) for pdf in ordered_player_dfs]}")
            return None
        
        combined_df['frame'] = np.arange(len(combined_df))
        combined_df['original_frame'] = combined_df['frame'].copy()

        combined_df = handle_null_values(combined_df)
        combined_df = update_boost_pad_timers(combined_df)

        if combined_df.empty:
            logging.error(f"No valid frames remaining after null/initial processing for {replay_file.name}")
            return None

        goal_events = []
        raw_goals = metadata.get('game', {}).get('goals', [])
        for g in raw_goals:
            goal_frame = g.get('frame')
            is_orange_goal = g.get('is_orange')
            goal_time_meta = g.get('time')

            if goal_frame is None or is_orange_goal is None:
                logging.warning(f"Incomplete goal data in metadata for {replay_file.name}: {g}. Skipping goal.")
                continue

            goal_time_to_use = None
            if goal_time_meta is not None:
                goal_time_to_use = float(goal_time_meta)
            elif 'time' in combined_df.columns and 0 <= goal_frame < len(combined_df):
                try:
                    goal_time_to_use = combined_df.iloc[goal_frame]['time']
                except (IndexError, KeyError):
                    logging.warning(f"Could not determine time for goal at frame {goal_frame} for {replay_file.name} from DF. Max index: {len(combined_df)-1}. Skipping goal.")
                    continue
            else:
                logging.warning(f"Could not determine time for goal at frame {goal_frame} (time col/frame missing or out of bounds). Skipping goal.")
                continue
            
            if goal_time_to_use is not None:
                 goal_events.append(GoalEvent(
                    frame=goal_frame,
                    time=goal_time_to_use,
                    team=1 if is_orange_goal else 0
                ))


        goal_label_cols = [f'team_{t}_goal_prev_{GOAL_ANTICIPATION_WINDOW_SECONDS}s' for t in [0, 1]]
        for col in goal_label_cols:
            combined_df[col] = 0

        for goal in goal_events:
            if 'time' not in combined_df.columns:
                logging.warning("Missing 'time' column, cannot label goal events.")
                break
            try:
                goal_time_float = float(goal.time)
                mask = (combined_df['time'] >= goal_time_float - GOAL_ANTICIPATION_WINDOW_SECONDS) & \
                       (combined_df['time'] < goal_time_float)
                combined_df.loc[mask, f'team_{goal.team}_goal_prev_{GOAL_ANTICIPATION_WINDOW_SECONDS}s'] = 1
            except ValueError:
                logging.warning(f"Invalid time value for goal: {goal.time}. Skipping this goal labeling.")


        if 'ball_hit_team_num' in combined_df.columns and 'original_frame' in combined_df.columns:
            combined_df = clean_post_goal_frames_using_ball_pos(
                combined_df, 
                [g.frame for g in goal_events]
            )
        if combined_df.empty:
            logging.warning(f"All data removed after post-goal cleaning for {replay_file.name}. Skipping.")
            return None

        analyzer_path = output_dir / "analyzer.json"
        if analyzer_path.exists() and 'original_frame' in combined_df.columns:
            with open(analyzer_path, "r", encoding="utf8") as f:
                analyzer_data = json.load(f)
            valid_ranges = [
                (p['start_frame'], p['end_frame']) 
                for p in analyzer_data.get('gameplay_periods', [])
                if 'start_frame' in p and 'end_frame' in p
            ]
            if valid_ranges:
                valid_mask = combined_df['original_frame'].apply(
                    lambda f_num: any(start <= f_num <= end for start, end in valid_ranges)
                )
                combined_df = combined_df[valid_mask].copy()
                if combined_df.empty:
                    logging.warning(f"No data remaining after filtering by gameplay_periods for {replay_file.name}. Skipping.")
                    return None
            else:
                logging.warning(f"No valid gameplay_periods found for {replay_file.name}. Keeping all frames.")
        else:
            logging.warning(f"analyzer.json not found or 'original_frame' missing. Skipping gameplay period filtering for {replay_file.name}.")

        original_row_count = len(combined_df)
        if POSITIVE_STATE_TARGET_HZ < 30 or NEGATIVE_STATE_TARGET_HZ < 30: # Check against assumed original 30Hz
            if original_row_count > 0:
                combined_df = downsample_data(
                    combined_df,
                    original_hz=30, 
                    event_columns=goal_label_cols
                )
                logging.info(f"Downsampled {replay_file.name} (Config: P@{POSITIVE_STATE_TARGET_HZ}Hz, N@{NEGATIVE_STATE_TARGET_HZ}Hz): "
                             f"{original_row_count} -> {len(combined_df)} rows.")
            else:
                 logging.info(f"Skipping downsampling for {replay_file.name} (0 rows before downsample).")
        else:
            logging.info(f"No downsampling for {replay_file.name} (target HZ not < 30Hz). Rows: {original_row_count}.")

        if combined_df.empty:
            logging.warning(f"No data remaining after downsampling for {replay_file.name}. Skipping.")
            return None
        
        pos_count = sum(combined_df[col].sum() for col in goal_label_cols if col in combined_df.columns)
        neg_count = len(combined_df) - pos_count
        imbalance_ratio = neg_count / max(1, pos_count) if pos_count > 0 else float('inf')
        logging.info(f"[{replay_file.name}] Class balance: {pos_count} pos vs {neg_count} neg (Ratio: {imbalance_ratio:.1f}:1)")
        with counts_lock:
            total_counts["positive"] += int(pos_count)
            total_counts["negative"] += int(neg_count)

        # Calculate player-to-ball distances (optimized assignment)
        new_dist_cols_data = {}
        ball_pos_cols_check = ['ball_pos_x', 'ball_pos_y', 'ball_pos_z']
        if all(col in combined_df.columns for col in ball_pos_cols_check):
            ball_positions = combined_df[ball_pos_cols_check].apply(pd.to_numeric, errors='coerce').fillna(0)
            for i in range(6):
                player_pos_cols_check = [f'p{i}_pos_x', f'p{i}_pos_y', f'p{i}_pos_z']
                if all(col in combined_df.columns for col in player_pos_cols_check):
                    player_positions = combined_df[player_pos_cols_check].apply(pd.to_numeric, errors='coerce').fillna(0)
                    diff_sq = (player_positions.values - ball_positions.values)**2
                    dist = np.sqrt(diff_sq.sum(axis=1))
                    new_dist_cols_data[f'p{i}_dist_to_ball'] = np.round(dist, 2)
                elif f'p{i}_pos_x' in combined_df.columns: # Player exists but maybe some pos component missing (unlikely after handle_nulls)
                    new_dist_cols_data[f'p{i}_dist_to_ball'] = MAX_MAP_DISTANCE
        else:
            logging.warning(f"Missing ball position columns for {replay_file.name} - filling dist_to_ball with MAX_MAP_DISTANCE.")
            for i in range(6):
                if f'p{i}_pos_x' in combined_df.columns: # Only if player data exists
                     new_dist_cols_data[f'p{i}_dist_to_ball'] = MAX_MAP_DISTANCE
        
        if new_dist_cols_data:
            combined_df = combined_df.assign(**new_dist_cols_data)
        
        cols_to_drop = [
            "original_frame", "frame", "time", "is_overtime",
            *[f'p{i}_quat_{c}' for i in range(6) for c in ['w','x','y','z']],
            *[f'p{i}_boost_pickup' for i in range(6)],
        ]
        extra_cols_to_drop_patterns = [f'p{i}_ball_dir_[xyz]' for i in range(6)]
        for pattern in extra_cols_to_drop_patterns:
            regex_pattern = pattern.replace('[xyz]', '([xyz])') # Simpler regex for fixed prefix
            cols_to_drop.extend([col for col in combined_df.columns if re.fullmatch(regex_pattern, col)])

        combined_df.drop(columns=cols_to_drop, errors='ignore', inplace=True)
        
        individual_csv_filename_only = INDIVIDUAL_REPLAY_CSV_FILENAME_FORMAT.format(stem=replay_file.stem) # Add .csv extension
        csv_path = individual_csv_output_path / individual_csv_filename_only # Use the passed directory
        
        combined_df.to_csv(csv_path, index=False)
        
        try:
            import shutil; shutil.rmtree(output_dir)
        except Exception as e:
            logging.warning(f"Couldn't delete temporary output directory {output_dir.name}: {str(e)}")
        
        processing_time = time.monotonic() - replay_start_time_mono
        logging.info(f"Successfully processed {replay_file.name} in {processing_time:.1f}s. Saved to {csv_path.name}")
        return combined_df

    except Exception as e:
        logging.exception(f"Critical failure processing {replay_file.name}: {str(e)}")
        if output_dir.exists():
            try:
                import shutil; shutil.rmtree(output_dir)
            except Exception as e_clean:
                logging.warning(f"Failed to cleanup {output_dir} after error: {e_clean}")
        return None

# ==================================================================
# Main Execution
# ==================================================================
def main():
    """Main execution function"""
    configure_logging() # Setup logging first
    
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

    # Update global filename formats (note: INDIVIDUAL_REPLAY_CSV_FILENAME_FORMAT is for the stem part only)
    INDIVIDUAL_REPLAY_CSV_FILENAME_FORMAT = f"{freq_str}_{time_window_str}_game_positions_{replay_group_name_sanitized}_{{stem}}" # .csv added later
    COMBINED_DATASET_CSV_FILENAME = f"dataset_{freq_str}_{time_window_str}_{replay_group_name_sanitized}.csv"
    PROCESSING_SUMMARY_FILENAME = f"processing_summary_{freq_str}_{time_window_str}_{replay_group_name_sanitized}.txt"
    
    logging.info(f"Individual CSVs will be named like: {INDIVIDUAL_REPLAY_CSV_FILENAME_FORMAT.format(stem='REPLAY_NAME')}.csv")
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
        future_to_replay = {
            executor.submit(process_replay, replay_file, individual_csvs_output_dir): replay_file
            for replay_file in replays
        }
        
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