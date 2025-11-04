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
from typing import Dict, List, Tuple, Optional
from threading import Lock

total_counts = {"positive": 0, "negative": 0}
counts_lock = Lock()

# ==================================================================
# Configuration
# ==================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
CARBALL_EXE = SCRIPT_DIR / "carball.exe"
PARENT_DIR = Path(r"E:\\Raw RL Esports Replays\\Day 3 Swiss Stage")
MAX_WORKERS = 4
TARGET_HZ = 5  # Target sampling frequency (5Hz)
# ==================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('replay_processor.log'),
        logging.StreamHandler()
    ]
)

def downsample_data(df: pd.DataFrame, original_hz: int = 30, target_hz: int = 5,
                   event_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Downsample dataframe while preserving event-labeled rows"""
    if event_columns is None:
        event_columns = []
    
    sampling_interval = int(original_hz / target_hz)
    positive_mask = pd.Series(False, index=df.index)
    for col in event_columns:
        if col in df.columns:
            positive_mask = positive_mask | (df[col] == 1)
    
    regular_samples = df[~positive_mask].iloc[::sampling_interval].index
    event_samples = df[positive_mask].index
    keep_indices = sorted(set(regular_samples).union(set(event_samples)))
    return df.loc[keep_indices].reset_index(drop=True)

def clean_post_goal_frames(df: pd.DataFrame, goal_frames: list) -> pd.DataFrame:
    """Remove frames between goals and subsequent kickoffs"""
    if 'ball_hit_team_num' not in df.columns:
        return df

    # Find all kickoffs after goals
    keep_mask = pd.Series(True, index=df.index)
    goal_frames = sorted(goal_frames)
    
    for goal_frame in goal_frames:
        # Find goal row
        goal_idx = df.index[df['original_frame'] == goal_frame].tolist()
        if not goal_idx:
            continue
        
        goal_idx = goal_idx[0]
        
        # Find next kickoff (where ball_hit_team_num becomes null)
        post_goal = df.iloc[goal_idx:]
        kickoff_idx = post_goal['ball_hit_team_num'].isnull().idxmax()
        
        if not pd.isna(kickoff_idx):
            # Remove frames between goal and kickoff (but keep kickoff)
            keep_mask.loc[goal_idx:kickoff_idx-1] = False

    return df[keep_mask].copy()

def find_replay_files(root_dir: Path) -> List[Path]:
    """Find all replay files with error handling"""
    try:
        return list(root_dir.rglob("*.replay"))
    except Exception as e:
        logging.error(f"Error finding replay files: {str(e)}")
        return []

def run_carball(replay_file: Path, output_dir: Path) -> bool:
    """Run carball.exe process"""
    command = [str(CARBALL_EXE), "parquet", "-i", str(replay_file), "-o", str(output_dir)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"carball.exe failed for {replay_file.name}:\n{e.stderr}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error running carball: {str(e)}")
        return False

def process_player_data(output_dir: Path, metadata: dict) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """Process player data with consistent ordering (p0-p5)"""
    # First validate and clean player data
    valid_players = []
    for player in metadata.get('players', []):
        if not all(k in player for k in ['is_orange', 'unique_id']):
            continue
        if not isinstance(player['is_orange'], bool):
            continue
        if player.get('unique_id') is None:
            continue
        valid_players.append(player)
    
    if len(valid_players) != 6:
        logging.warning(f"Expected 6 players, found {len(valid_players)}")
    
    # Sort players: blue team first (is_orange=False), then orange team, sorted by unique_id
    players = sorted(valid_players,
                    key=lambda p: (p['is_orange'], str(p['unique_id'])))  # Convert unique_id to string for safe comparison
    
    player_dfs = {}
    player_columns = []
    for idx, player in enumerate(players):  # Now guaranteed 0-5 order
        player_id = str(player['unique_id'])
        player_file = output_dir / f"player_{player_id}.parquet"
        
        if not player_file.exists():
            logging.warning(f"Missing player file for {player_id}")
            continue
            
        try:
            player_df = pd.read_parquet(player_file)
            # Select and rename columns with consistent player index
            cols = ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'boost_amount']
            player_df = player_df[cols]
            player_df.columns = [f"p{idx}_{col}" for col in cols]
            player_df[f"p{idx}_team"] = 1 if player['is_orange'] else 0
            player_dfs[player_id] = player_df
            player_columns.extend([f"p{idx}_{col}" for col in cols] + [f"p{idx}_team"])
        except Exception as e:
            logging.error(f"Error processing player {player_id}: {str(e)}")
            
    return player_dfs, player_columns

def process_replay(replay_file: Path):
    """Process a single replay file with consistent player ordering"""
    output_folder_name = f"Output-{replay_file.stem}"
    output_dir = replay_file.parent / output_folder_name
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Run carball
    if not run_carball(replay_file, output_dir):
        return

    try:
        # Step 2: Load metadata
        with open(output_dir / 'metadata.json', 'r', encoding='utf8') as f:
            metadata = json.load(f)
        
        # Step 3: Process player data with consistent ordering
        player_dfs, player_columns = process_player_data(output_dir, metadata)
        
        # Step 4: Process goal data
        goal_events = metadata["game"]["goals"]
        goal_data = [(g["frame"], 0 if not g["is_orange"] else 1) for g in goal_events]
        
        # Step 5: Load and process game/ball data
        game_df = pd.read_parquet(output_dir / '__game.parquet').drop(
            columns=['delta', 'replicated_game_state_time_remaining', 'ball_has_been_hit'],
            errors='ignore'
        )
        
        ball_df = pd.read_parquet(output_dir / '__ball.parquet').add_prefix('ball_').drop(
            columns=['ball_quat_w', 'ball_quat_x', 'ball_quat_y', 'ball_quat_z',
                    'ball_ang_vel_x', 'ball_ang_vel_y', 'ball_ang_vel_z', 
                    'ball_is_sleeping', 'ball_has_been_hit'],
            errors='ignore'
        )

        # Step 6: Combine all data
        combined_df = pd.concat([game_df] + list(player_dfs.values()) + [ball_df], axis=1).round(2)
        
        # 1. Ensure frame tracking exists
        combined_df['frame'] = np.arange(len(combined_df))
        combined_df['original_frame'] = combined_df['frame'].copy()

        # 2. Load goals from metadata
        with open(output_dir / 'metadata.json', 'r', encoding='utf-8') as f:
            goal_data = [
                (g['frame'], 1 if g['is_orange'] else 0)
                for g in json.load(f).get('game', {}).get('goals', [])
                if 'frame' in g and 'is_orange' in g
            ]

        # 3. Initialize goal labels
        goal_columns = [f'team_{t}_goal_prev_5s' for t in [0, 1]]
        for col in goal_columns:
            combined_df[col] = 0

        # 4. Label goal periods (BEFORE filtering)
        for goal_frame, team in goal_data:
            goal_rows = combined_df[combined_df['original_frame'] == goal_frame]
            if not goal_rows.empty:
                goal_time = goal_rows.iloc[0]['time']
                mask = (combined_df['time'] >= goal_time - 5) & (combined_df['time'] < goal_time)
                combined_df.loc[mask, f'team_{team}_goal_prev_5s'] = 1

        # 5. Clean post-goal frames (NEW)
        if 'ball_hit_team_num' in combined_df.columns:
            combined_df = clean_post_goal_frames(combined_df, [g[0] for g in goal_data])

        # 6. Filter valid gameplay frames (AFTER labeling and cleaning)
        with open(output_dir / "analyzer.json", "r", encoding="utf8") as f:
            valid_ranges = [(p['start_frame'], p['end_frame']) for p in json.load(f).get('gameplay_periods', [])]
        
        valid_mask = combined_df['original_frame'].apply(
            lambda f: any(start <= f <= end for start, end in valid_ranges)
        )
        combined_df = combined_df[valid_mask].copy()

        # Step 10: Downsample if needed
        original_row_count = len(combined_df)
        if TARGET_HZ < 30:
            combined_df = downsample_data(
                combined_df,
                original_hz=30,
                target_hz=TARGET_HZ,
                event_columns=goal_columns
            )
            logging.info(f"Downsampled {replay_file.name} from 30Hz to {TARGET_HZ}Hz: "
                        f"{original_row_count} to {len(combined_df)} rows")
        
        # Step 11: Log class balance
        pos_count = sum(combined_df[col].sum() for col in goal_columns)
        neg_count = len(combined_df) - pos_count
        imbalance_ratio = neg_count / max(1, pos_count)

        logging.info(f"[{replay_file.name}] Class balance: {pos_count} positive vs {neg_count} negative samples "
                    f"(ratio: {imbalance_ratio:.1f}:1)")

        # Update global tally
        with counts_lock:
            total_counts["positive"] += int(pos_count)
            total_counts["negative"] += int(neg_count)
        
        # Step 12: Drop the columns that are not required after usage
        combined_df.drop(
            columns=["original_frame", "frame", "time", "seconds_remaining", "is_overtime", 
            "p0_vel_x", "p0_vel_y", "p0_vel_z", "p0_boost_amount",
            "p1_vel_x", "p1_vel_y", "p1_vel_z", "p1_boost_amount",
            "p2_vel_x", "p2_vel_y", "p2_vel_z", "p2_boost_amount",
            "p3_vel_x", "p3_vel_y", "p3_vel_z", "p3_boost_amount",
            "p4_vel_x", "p4_vel_y", "p4_vel_z", "p4_boost_amount",
            "p5_vel_x", "p5_vel_y", "p5_vel_z", "p5_boost_amount",
            "ball_pos_x", "ball_pos_y", "ball_pos_z", "ball_vel_x", "ball_vel_y", "ball_vel_z", "ball_hit_team_num"], 
            errors='ignore', inplace=True
        )
        
        csv_path = output_dir / f"game_positions_{replay_file.stem}.csv"
        combined_df.to_csv(csv_path, index=False)
        
        # Step 13: Cleanup
        for ext in ('*.parquet', '*.json'):
            for file in output_dir.glob(ext):
                try:
                    file.unlink()
                except Exception as e:
                    logging.warning(f"Couldn't delete {file.name}: {str(e)}")
        
        # Step 13: Return to combined dataframe
        return combined_df

    except Exception as e:
        logging.error(f"Failed to process {replay_file.name}: {str(e)}")

def main():
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
    replay_stats = []  # For detailed summary per replay
    failed_replays = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_replay, replay): replay for replay in replays}
        
        for future in as_completed(futures):
            replay = futures[future]
            try:
                result_df = future.result()
                if result_df is not None:
                    all_dfs.append(result_df)
                    replay_stats.append({
                        'replay_name': replay.name,
                        'rows': len(result_df)
                    })
                else:
                    failed_replays.append(replay.name)
            except Exception as e:
                failed_replays.append(replay.name)
                logging.error(f"Thread error for {replay.name}: {str(e)}")

    # Combine and save final CSV
    summary_lines = []
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        output_csv = PARENT_DIR / "starter_all_replays_combined.csv"
        final_df.to_csv(output_csv, index=False)
        logging.info(f"Saved combined CSV: {output_csv}")

        total_rows = len(final_df)
        summary_lines.append(f"Total combined rows: {total_rows}")
    else:
        logging.warning("No valid replay data to combine.")
        summary_lines.append("No valid dataframes were combined.")

    # Timing and counts
    total_time = time.time() - start_time
    minutes, seconds = divmod(total_time, 60)

    summary_lines.insert(0, f"Total replays found: {len(replays)}")
    summary_lines.append(f"Successfully processed: {len(replay_stats)}")
    summary_lines.append(f"Failed replays: {len(failed_replays)}")
    summary_lines.append(f"Total processing time: {int(minutes)}m {seconds:.1f}s")

    if failed_replays:
        summary_lines.append("\nFailed replays:")
        summary_lines.extend(f" - {name}" for name in failed_replays)

    if replay_stats:
        summary_lines.append("\nReplay-wise row counts:")
        summary_lines.extend(f" - {stat['replay_name']}: {stat['rows']} rows" for stat in replay_stats)

    # Write to summary log
    summary_path = PARENT_DIR / "processing_summary.log"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(summary_lines))

    logging.info(f"Summary written to: {summary_path}")
    
    # Final aggregated class balance
    with counts_lock:
        pos_total = total_counts["positive"]
        neg_total = total_counts["negative"]
        total = pos_total + neg_total
        imbalance_ratio = neg_total / max(1, pos_total)

    logging.info(f"[TOTAL] Class balance across all replays: {pos_total} positive vs {neg_total} negative samples "
                f"(ratio: {imbalance_ratio:.1f}:1, total: {total})")


if __name__ == "__main__":
    main()