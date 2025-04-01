import subprocess
import pandas as pd
import json
from pathlib import Path
import os
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# ==================================================================
# Configuration (Update path for carball.exe!)
# Configuration (Update the path for the parent replay group!)
# ==================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
CARBALL_EXE = SCRIPT_DIR / "carball.exe"

if not CARBALL_EXE.exists():
    print(f"Error: carball.exe not found at {CARBALL_EXE}")
    sys.exit(1)

PARENT_DIR = Path(r"E:\\Raw RL Esports Replays\\Day 3 Swiss Stage")  # Root directory for replay files

# ==================================================================

def find_replay_files(root_dir: Path):
    """Recursively finds all .replay files under root_dir"""
    return list(root_dir.rglob("*.replay"))

def process_replay(replay_file: Path):
    """Processes a single replay file with dedicated output folder"""
    output_folder_name = f"Output-{replay_file.stem}"
    output_dir = replay_file.parent / output_folder_name
    output_dir.mkdir(exist_ok=True)
    
    command = [
        str(CARBALL_EXE), "parquet", "-i", str(replay_file), "-o", str(output_dir)
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"\U0001F6D2 Failed to process {replay_file.name}:{result.stderr}")
        return

    try:
        with open(output_dir / 'metadata.json', 'r', encoding='utf8') as f:
            metadata = json.load(f)
        
        player_ids = [p['unique_id'] for p in metadata['players']]
        player_teams = {p['unique_id']: (0 if not p['is_orange'] else 1) for p in metadata['players']}

        # Extract goal events and scoring teams
        goal_events = metadata["game"]["goals"]
        goal_frames = []
        goal_teams = []

        for goal in goal_events:
            goal_frames.append(goal["frame"])  # Frame index in unfiltered data
            goal_teams.append(0 if not goal["is_orange"] else 1)  # Team 0 (Blue) if is_orange=False, else Team 1 (Orange)

        print(f"📌 Found {len(goal_frames)} goals!")

        game_df = pd.read_parquet(output_dir / '__game.parquet')
        ball_df = pd.read_parquet(output_dir / '__ball.parquet').add_prefix('ball_')

        # Drop unnecessary columns from game_df
        game_df.drop(columns=[
            'delta', 'replicated_game_state_time_remaining',
            'ball_has_been_hit'
        ], errors='ignore', inplace=True)
        
        # Drop unnecessary columns from ball_df
        ball_df.drop(columns=[
            'ball_quat_w', 'ball_quat_x', 'ball_quat_y', 'ball_quat~_z',
            'ball_ang_vel_x', 'ball_ang_vel_y', 'ball_ang_vel_z', 'ball_is_sleeping', 
            'ball_has_been_hit'
        ], errors='ignore', inplace=True)

        player_dfs = {}
        player_index = 0
        for player_id in player_ids:
            player_file = output_dir / f"player_{player_id}.parquet"
            if player_file.exists():
                player_df = pd.read_parquet(player_file)
                player_df = player_df[['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'boost_amount']]
                player_df.columns = [f"p{player_index}_{col}" for col in player_df.columns]
                player_df[f"p{player_index}_team"] = player_teams[player_id]  # Add team column
                player_dfs[player_id] = player_df
                player_index += 1

        combined_df = pd.concat([game_df] + list(player_dfs.values()) + [ball_df], axis=1)
        combined_df = combined_df.round(2)
        # combined_df.sort_values(by=['time'], inplace=True)

        # Load analyzer.json to filter frames within valid gameplay periods
        with open(output_dir / "analyzer.json", "r", encoding="utf8") as f:
            analyzer_data = json.load(f)

        gameplay_periods = analyzer_data["gameplay_periods"]
        valid_frame_ranges = [(p["start_frame"], p["end_frame"]) for p in gameplay_periods]

        # Assign frame numbers based on original unfiltered dataset (row index before filtering)
        combined_df["frame"] = np.arange(len(combined_df))  # Add a frame column

        # Remove frames outside valid gameplay periods
        valid_frames_mask = combined_df["frame"].apply(
            lambda frame: any(start <= frame <= end for start, end in valid_frame_ranges)
        )
        combined_df = combined_df[valid_frames_mask].reset_index(drop=True)
        print(f"🧹 Filtered dataset to {len(combined_df)} valid frames!")

        # Label previous 5 seconds for each goal
        combined_df["team_0_goal_prev_5s"] = 0
        combined_df["team_1_goal_prev_5s"] = 0

        for goal_frame, team in zip(goal_frames, goal_teams):
            # Find the corresponding row index in the filtered dataset
            goal_row_idx = combined_df.index[combined_df["frame"] == goal_frame].tolist()
            
            if len(goal_row_idx) == 0:
                continue  # Skip if goal frame is missing due to filtering
            
            goal_row_idx = goal_row_idx[0]  # Get the single matching row index

            # Get the goal time from the dataset
            goal_time = combined_df.loc[goal_row_idx, "time"]

            # Find rows where time is within the previous 5 seconds
            mask = (combined_df["time"] >= goal_time - 5) & (combined_df["time"] < goal_time)
            num_rows = mask.sum()
            print(f"✅ {num_rows} rows labeled for Team {team} (before goal at {goal_time:.2f}s)")

            combined_df.loc[mask, f"team_{team}_goal_prev_5s"] = 1

        # Drop the added frame column to keep the output clean
        combined_df.drop(columns=["frame"], inplace=True)

        ### 🚨 Step 2: Conflict Resolution ###
        conflict_mask = (combined_df["team_0_goal_prev_5s"] == 1) & (combined_df["team_1_goal_prev_5s"] == 1)

        if conflict_mask.sum() > 0:
            print(f"⚠️ Found {conflict_mask.sum()} conflicting labels!")

            conflict_rows = combined_df[conflict_mask]

            # Group conflicting rows by the goal timestamp
            for conflict_time in conflict_rows["time"].unique():
                scoring_teams = [team for time, team in goal_timestamps.items() if conflict_time - 5 <= time <= conflict_time]
                
                if len(scoring_teams) > 1:
                    # More than one team scored within 5 seconds → determine which was first
                    scoring_teams.sort(key=lambda team: list(goal_timestamps.keys())[list(goal_timestamps.values()).index(team)])

                    first_team = scoring_teams[0]
                    second_team = scoring_teams[1]

                    # Ensure first team's label remains, and second team's label shifts
                    print(f"🔄 Adjusting Team {second_team}'s labels to start after Team {first_team}'s period ends.")

                    # Remove second team's overlapping labels
                    combined_df.loc[conflict_mask & (combined_df["time"] < (conflict_time - 5)), f"team_{second_team}_goal_prev_5s"] = 0

        print("✅ Conflict resolution complete!")


        csv_name = f"altered_game_positions_{replay_file.stem}.csv"
        combined_df.to_csv(output_dir / csv_name, index=False)
        print(f"✅ Success: {replay_file.name} → {csv_name}")

        # Cleanup intermediate files
        for file in output_dir.glob("*.parquet"):
            file.unlink()
        for file in output_dir.glob("*.json"):
            file.unlink()

    except Exception as e:
        print(f"❌ Error processing {replay_file.name}: {str(e)}")


def main():
    start_time = time.time()
    replays = find_replay_files(PARENT_DIR)
    print(f"🔍 Found {len(replays)} replay files")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_replay, replay): replay for replay in replays}
        for future in as_completed(futures):
            replay = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"❌ Error processing {replay.name}: {str(e)}")

    total_sec = time.time() - start_time
    mins, secs = divmod(total_sec, 60)
    print(f"\n⏱ Total processing time: {int(mins)}m {secs:.1f}s")

if __name__ == "__main__":
    main()
