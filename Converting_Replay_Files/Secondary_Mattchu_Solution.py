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

# ==================================================================
# Global Configuration
# ==================================================================
# Using global variables for configuration that would typically come from a config file
SCRIPT_DIR = Path(__file__).resolve().parent
CARBALL_EXE = SCRIPT_DIR / "carball.exe"  # Path to carball executable
PARENT_DIR = Path(r"E:\\Raw RL Esports Replays\\Day 3 Swiss Stage")  # Root directory containing replays
MAX_WORKERS = 4  # Maximum parallel threads for processing
TARGET_HZ = 5  # Target sampling frequency in Hz (downsamples from 30Hz)
LOG_LEVEL = logging.INFO  # Logging verbosity

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
        level=logging.INFO,
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
                   target_hz: int = 5,
                   event_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Downsample dataframe while preserving event-labeled rows.
    
    Args:
        df: Input DataFrame with time-series data
        original_hz: Original sampling frequency (default 30Hz for Rocket League)
        target_hz: Target sampling frequency
        event_columns: List of column names marking important events to preserve
        
    Returns:
        Downsampled DataFrame with event rows preserved
    """
    if event_columns is None:
        event_columns = []
    
    # Calculate sampling interval
    sampling_interval = int(original_hz / target_hz)
    if sampling_interval < 1:
        return df  # No downsampling needed
        
    # Create mask for event rows we want to preserve
    positive_mask = pd.Series(False, index=df.index)
    for col in event_columns:
        if col in df.columns:
            positive_mask = positive_mask | (df[col] == 1)
    
    # Select regular samples (non-event rows) and all event rows
    regular_samples = df[~positive_mask].iloc[::sampling_interval].index
    event_samples = df[positive_mask].index
    
    # Combine and sort the indices to keep
    keep_indices = sorted(set(regular_samples).union(set(event_samples)))
    return df.loc[keep_indices].reset_index(drop=True)

def clean_post_goal_frames(df: pd.DataFrame, 
                          goal_frames: List[int]) -> pd.DataFrame:
    """
    Remove frames between goals and subsequent kickoffs to eliminate dead time.
    
    Args:
        df: DataFrame containing game frames
        goal_frames: List of frame numbers where goals occurred
        
    Returns:
        Filtered DataFrame with post-goal frames removed
    """
    if 'ball_hit_team_num' not in df.columns:
        return df  # Can't identify kickoffs without this column

    # Initialize mask to keep all frames by default
    keep_mask = pd.Series(True, index=df.index)
    goal_frames = sorted(goal_frames)  # Process goals in chronological order
    
    for goal_frame in goal_frames:
        # Find the exact row where the goal occurred
        goal_idx = df.index[df['original_frame'] == goal_frame].tolist()
        if not goal_idx:
            continue  # Goal frame not found in this DataFrame
            
        goal_idx = goal_idx[0]  # Get first occurrence
        
        # Find next kickoff (where ball_hit_team_num becomes null)
        post_goal = df.iloc[goal_idx:]
        kickoff_idx = post_goal['ball_hit_team_num'].isnull().idxmax()
        
        if not pd.isna(kickoff_idx):
            # Remove frames between goal and kickoff (but keep the kickoff frame)
            keep_mask.loc[goal_idx:kickoff_idx-1] = False

    return df[keep_mask].copy()

def calculate_distance(player_pos: Tuple[float, float, float], 
                      ball_pos: Tuple[float, float, float]) -> float:
    """
    Calculate Euclidean distance between player and ball positions in 3D space.
    
    Args:
        player_pos: Tuple of (x, y, z) player coordinates
        ball_pos: Tuple of (x, y, z) ball coordinates
        
    Returns:
        Distance between player and ball
    """
    return np.sqrt(
        (player_pos[0] - ball_pos[0])**2 +
        (player_pos[1] - ball_pos[1])**2 +
        (player_pos[2] - ball_pos[2])**2
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
    return (x/norm, y/norm, z/norm)

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
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"carball.exe failed for {replay_file.name}:\n{e.stderr}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error running carball: {str(e)}")
        return False

def process_player_data(output_dir: Path, 
                       metadata: dict) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Process player data from carball output with consistent team/player ordering.
    
    Args:
        output_dir: Directory containing carball output files
        metadata: Parsed metadata from carball
        
    Returns:
        Tuple of (player DataFrames dictionary, list of player column names)
    """
    # Validate and clean player metadata
    valid_players = []
    for player in metadata.get('players', []):
        # Skip players missing critical fields or with invalid data
        if not all(k in player for k in ['is_orange', 'unique_id']):
            continue
        if not isinstance(player['is_orange'], bool):
            continue
        if player.get('unique_id') is None:
            continue
        valid_players.append(player)
    
    # Warn if we don't have exactly 6 players (standard for Rocket League)
    if len(valid_players) != 6:
        logging.warning(f"Expected 6 players, found {len(valid_players)}")
    
    # Sort players: blue team first (is_orange=False), then orange team
    players = sorted(valid_players,
                    key=lambda p: (p['is_orange'], str(p['unique_id'])))
    
    player_dfs = {}
    player_columns = []
    
    for idx, player in enumerate(players):
        player_id = str(player['unique_id'])
        player_file = output_dir / f"player_{player_id}.parquet"
        
        if not player_file.exists():
            continue
            
        try:
            player_df = pd.read_parquet(player_file)
            
            # Standard columns we'll keep
            base_cols = ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'boost_amount']
            
            # Add quaternion columns with proper naming
            quat_cols = {
                'quat_w': f'p{idx}_quat_w',
                'quat_x': f'p{idx}_quat_x',
                'quat_y': f'p{idx}_quat_y',
                'quat_z': f'p{idx}_quat_z'
            }
            
            # Vectorized forward vector calculation
            qw, qx, qy, qz = player_df['quat_w'], player_df['quat_x'], player_df['quat_y'], player_df['quat_z']
            x = 1 - 2 * (qy**2 + qz**2)
            y = 2 * (qx*qy + qw*qz)
            z = 2 * (qx*qz - qw*qy)
            norm = np.sqrt(x**2 + y**2 + z**2)
            
            # Create new DataFrame with properly named columns
            processed_df = pd.DataFrame({
                **{f'p{idx}_{col}': player_df[col] for col in base_cols},
                **quat_cols,
                f'p{idx}_forward_x': x/norm,
                f'p{idx}_forward_y': y/norm,
                f'p{idx}_forward_z': z/norm,
                f'p{idx}_team': 1 if player['is_orange'] else 0
            })
            
            player_dfs[player_id] = processed_df
            player_columns.extend(processed_df.columns.tolist())
            
        except Exception as e:
            logging.error(f"Error processing player {player_id}: {str(e)}")
            
    return player_dfs, player_columns

def process_replay(replay_file: Path) -> Optional[pd.DataFrame]:
    """
    Full processing pipeline for a single replay file.
    
    Args:
        replay_file: Path to the replay file to process
        
    Returns:
        Processed DataFrame if successful, None otherwise
    """
    replay_start = time.time()
    output_folder_name = f"Output-{replay_file.stem}"
    output_dir = replay_file.parent / output_folder_name
    output_dir.mkdir(exist_ok=True)
    
    logging.info(f"Processing {replay_file.name}...")
    
    # Step 1: Run carball to extract raw data
    if not run_carball(replay_file, output_dir):
        return None

    try:
        # Step 2: Load metadata
        with open(output_dir / 'metadata.json', 'r', encoding='utf8') as f:
            metadata = json.load(f)
        
        # Step 3: Process player data with consistent ordering
        player_dfs, player_columns = process_player_data(output_dir, metadata)
        
        # Step 4: Load and process game data
        game_df = pd.read_parquet(output_dir / '__game.parquet').drop(
            columns=['delta', 'replicated_game_state_time_remaining', 'ball_has_been_hit'],
            errors='ignore'
        )
        
        # Step 5: Load and process ball data
        ball_df = pd.read_parquet(output_dir / '__ball.parquet').add_prefix('ball_').drop(
            columns=['ball_quat_w', 'ball_quat_x', 'ball_quat_y', 'ball_quat_z',
                    'ball_ang_vel_x', 'ball_ang_vel_y', 'ball_ang_vel_z', 
                    'ball_is_sleeping', 'ball_has_been_hit'],
            errors='ignore'
        )

        # Step 6: Combine all data sources
        combined_df = pd.concat(
            [game_df] + list(player_dfs.values()) + [ball_df], 
            axis=1
        ).round(2)  # Reduce precision to save space
        
        # Add frame tracking columns
        combined_df['frame'] = np.arange(len(combined_df))
        combined_df['original_frame'] = combined_df['frame'].copy()

        # Extract goal events from metadata
        goal_events = [
            GoalEvent(
                frame=g['frame'],
                time=combined_df.loc[combined_df['original_frame'] == g['frame'], 'time'].values[0],
                team=1 if g['is_orange'] else 0
            )
            for g in metadata.get('game', {}).get('goals', [])
            if 'frame' in g and 'is_orange' in g
        ]

        # Initialize goal label columns
        goal_columns = [f'team_{t}_goal_prev_5s' for t in [0, 1]]
        for col in goal_columns:
            combined_df[col] = 0

        # Label frames in the 5 seconds before each goal
        for goal in goal_events:
            mask = (combined_df['time'] >= goal.time - 5) & (combined_df['time'] < goal.time)
            combined_df.loc[mask, f'team_{goal.team}_goal_prev_5s'] = 1

        # Clean post-goal frames (remove dead time between goals and kickoffs)
        if 'ball_hit_team_num' in combined_df.columns:
            combined_df = clean_post_goal_frames(
                combined_df, 
                [g.frame for g in goal_events]
            )

        # Filter to only valid gameplay periods (exclude menus, replays, etc.)
        with open(output_dir / "analyzer.json", "r", encoding="utf8") as f:
            valid_ranges = [
                (p['start_frame'], p['end_frame']) 
                for p in json.load(f).get('gameplay_periods', [])
            ]
        
        valid_mask = combined_df['original_frame'].apply(
            lambda f: any(start <= f <= end for start, end in valid_ranges)
        )
        combined_df = combined_df[valid_mask].copy()

        # Downsample if target frequency is lower than original
        original_row_count = len(combined_df)
        if TARGET_HZ < 30:
            combined_df = downsample_data(
                combined_df,
                original_hz=30,
                target_hz=TARGET_HZ,
                event_columns=goal_columns
            )
            logging.info(f"Downsampled {replay_file.name} from 30Hz to {TARGET_HZ}Hz: "
                        f"{original_row_count} into {len(combined_df)} rows")
        
        # Calculate and log class balance for this replay
        pos_count = sum(combined_df[col].sum() for col in goal_columns)
        neg_count = len(combined_df) - pos_count
        imbalance_ratio = neg_count / max(1, pos_count)

        logging.info(f"[{replay_file.name}] Class balance: {pos_count} positive vs {neg_count} negative samples "
                    f"(ratio: {imbalance_ratio:.1f}:1)")

        # Update global counters
        with counts_lock:
            total_counts["positive"] += int(pos_count)
            total_counts["negative"] += int(neg_count)

        # Calculate player-to-ball distances
        ball_pos_cols = ['ball_pos_x', 'ball_pos_y', 'ball_pos_z']
        if all(col in combined_df.columns for col in ball_pos_cols):
            for i in range(6):  # For each player (p0-p5)
                player_pos_cols = [f'p{i}_pos_x', f'p{i}_pos_y', f'p{i}_pos_z']
                if all(col in combined_df.columns for col in player_pos_cols):
                    combined_df[f'p{i}_dist_to_ball'] = combined_df.apply(
                        lambda row: calculate_distance(
                            (row[player_pos_cols[0]], row[player_pos_cols[1]], row[player_pos_cols[2]]),
                            (row[ball_pos_cols[0]], row[ball_pos_cols[1]], row[ball_pos_cols[2]])
                        ),
                        axis=1
                    ).round(2)  # Round to 2 decimal places
        else:
            logging.warning("Missing ball position columns - skipping distance calculations")
        
        # Remove temporary columns no longer needed
        combined_df.drop(
            columns=[
                "original_frame", "frame", "time", "is_overtime"
                #"seconds_remaining", 
                #*[f"p{i}_{attr}" for i in range(6) for attr in ["vel_x", "vel_y", "vel_z", "boost_amount"]],
                #"ball_pos_x", "ball_pos_y", "ball_pos_z", 
                #"ball_vel_x", "ball_vel_y", "ball_vel_z", 
                #"ball_hit_team_num"
            ], 
            errors='ignore', 
            inplace=True
        )

        combined_df.drop(
            columns=[f'p{i}_ball_dir_{ax}' for i in range(6) for ax in ['x','y','z']],
            errors='ignore',
            inplace=True
        )

        combined_df.drop(columns=[f'p{i}_quat_{c}' for i in range(6) for c in ['w','x','y','z']], 
            errors='ignore'
        )
        
        # Save processed data to CSV
        csv_path = output_dir / f"game_positions_{replay_file.stem}.csv"
        combined_df.to_csv(csv_path, index=False)
        
        # Cleanup temporary files
        for ext in ('*.parquet', '*.json'):
            for file in output_dir.glob(ext):
                try:
                    file.unlink()
                except Exception as e:
                    logging.warning(f"Couldn't delete {file.name}: {str(e)}")
        
        logging.info(f"Successfully processed {replay_file.name} in {time.time() - replay_start:.1f}s")
        return combined_df

    except Exception as e:
        logging.error(f"Failed to process {replay_file.name}: {str(e)}")
        return None

# ==================================================================
# Main Execution
# ==================================================================
def main():
    """Main execution function"""
    configure_logging()
    
    # Validate carball executable exists
    if not CARBALL_EXE.exists():
        logging.critical(f"carball.exe not found at {CARBALL_EXE}")
        sys.exit(1)

    start_time = time.time()
    replays = find_replay_files(PARENT_DIR)

    if not replays:
        logging.warning("No replay files found!")
        return

    logging.info(f"Found {len(replays)} replay files (target: {TARGET_HZ}Hz)")

    all_dfs = []
    replay_stats = []
    failed_replays = []

    # Process replays in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_replay, replay): replay for replay in replays}
        
        for future in as_completed(futures):
            replay = futures[future]
            try:
                result_df = future.result()
                if result_df is not None:
                    all_dfs.append(result_df)
                    replay_stats.append(ReplayStats(
                        replay_name=replay.name,
                        rows=len(result_df),
                        processing_time=time.time() - start_time,
                        success=True
                    ))
                else:
                    failed_replays.append(replay.name)
            except Exception as e:
                failed_replays.append(replay.name)
                logging.error(f"Thread error for {replay.name}: {str(e)}")

    # Combine and save all processed data
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        output_csv = PARENT_DIR / "secondary_all_replays_combined.csv"
        final_df.to_csv(output_csv, index=False)
        logging.info(f"Saved combined CSV: {output_csv}")

    # Generate summary report
    generate_summary_report(
        replays=replays,
        replay_stats=replay_stats,
        failed_replays=failed_replays,
        start_time=start_time,
        output_dir=PARENT_DIR
    )

def generate_summary_report(replays: List[Path],
                          replay_stats: List[ReplayStats],
                          failed_replays: List[str],
                          start_time: float,
                          output_dir: Path):
    """
    Generate a comprehensive processing summary report.
    
    Args:
        replays: List of all replay paths found
        replay_stats: List of processing statistics for successful replays
        failed_replays: List of failed replay names
        start_time: Start time of processing
        output_dir: Directory to save the report
    """
    summary_lines = []
    total_time = time.time() - start_time
    minutes, seconds = divmod(total_time, 60)

    # Basic statistics
    summary_lines.append(f"Total replays found: {len(replays)}")
    summary_lines.append(f"Successfully processed: {len(replay_stats)}")
    summary_lines.append(f"Failed replays: {len(failed_replays)}")
    summary_lines.append(f"Total processing time: {int(minutes)}m {seconds:.1f}s")

    # Failed replays details
    if failed_replays:
        summary_lines.append("\nFailed replays:")
        summary_lines.extend(f" - {name}" for name in failed_replays)

    # Successful replay details
    if replay_stats:
        summary_lines.append("\nReplay processing details:")
        summary_lines.append(f"{'Replay Name':<50}{'Rows':>10}{'Time (s)':>10}")
        for stat in replay_stats:
            summary_lines.append(
                f"{stat.replay_name:<50}{stat.rows:>10}{stat.processing_time:>10.1f}"
            )

    # Final class balance
    with counts_lock:
        pos_total = total_counts["positive"]
        neg_total = total_counts["negative"]
        total = pos_total + neg_total
        imbalance_ratio = neg_total / max(1, pos_total)

    summary_lines.append("\nAggregate Class Balance:")
    summary_lines.append(f"Positive samples (goal imminent): {pos_total}")
    summary_lines.append(f"Negative samples: {neg_total}")
    summary_lines.append(f"Total samples: {total}")
    summary_lines.append(f"Class imbalance ratio: {imbalance_ratio:.1f}:1")

    # Write summary to file
    summary_path = output_dir / "processing_summary.log"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(summary_lines))

    logging.info(f"Summary written to: {summary_path}")

if __name__ == "__main__":
    main()