import subprocess
import pandas as pd
import json
from pathlib import Path
import os
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==================================================================
# Configuration (Update path for carball.exe!)
# Configuration (Update the path for the parent replay group!)
# ==================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
CARBALL_EXE = SCRIPT_DIR / "carball.exe"

if not CARBALL_EXE.exists():
    print(f"Error: carball.exe not found at {CARBALL_EXE}")
    sys.exit(1)

PARENT_DIR = Path(r"E:\\RL Esports Replays\\Swiss\\Round 1\\AM vs GG")  # Root directory for replay files

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
        print(f"🚨 Failed to process {replay_file.name}:{result.stderr}")
        return

    try:
        with open(output_dir / 'metadata.json', 'r', encoding='utf8') as f:
            metadata = json.load(f)
        
        player_ids = [p['unique_id'] for p in metadata['players']]
        player_names = {p['unique_id']: p['name'] for p in metadata['players']}
        player_teams = {p['unique_id']: (0 if not p['is_orange'] else 1) for p in metadata['players']}

        game_df = pd.read_parquet(output_dir / '__game.parquet')
        ball_df = pd.read_parquet(output_dir / '__ball.parquet').add_prefix('ball_')

        player_dfs = {}
        for player_id in player_ids:
            player_file = output_dir / f"player_{player_id}.parquet"
            if player_file.exists():
                player_df = pd.read_parquet(player_file)
                player_df = player_df[['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'boost']]
                player_df.columns = [f"P{idx+1}_{col}" for idx, col in enumerate(player_df.columns)]
                player_dfs[player_id] = player_df

        combined_df = pd.concat([game_df] + list(player_dfs.values()) + [ball_df], axis=1)
        combined_df.sort_values(by=['time'], inplace=True)

        csv_name = f"game_positions_{replay_file.stem}.csv"
        combined_df.to_csv(output_dir / csv_name, index=False)
        print(f"✅ Success: {replay_file.name} → {csv_name}")

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
