import subprocess
import pandas as pd
import json
from pathlib import Path
import os
import time

# ==================================================================
# Configuration (Update these paths!)
# ==================================================================
CARBALL_EXE = Path(r"C:\\Users\\Arslan\\Desktop\\MS-Thesis-Oguz-Arslan\\Converting_Replay_Files\\carball.exe")
PARENT_DIR = Path(r"E:\\RL Esports Replays")  # Root directory to scan for .replay files
# ==================================================================

def find_replay_files(root_dir: Path):
    """Recursively finds all .replay files under root_dir"""
    return list(root_dir.rglob("*.replay"))

def process_replay(replay_file: Path):
    """Processes a single replay file with dedicated output folder"""
    # Create unique output folder using replay filename
    output_folder_name = f"Output-{replay_file.stem}"
    output_dir = replay_file.parent / output_folder_name
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Run carball.exe
    command = [
        str(CARBALL_EXE),
        "parquet",
        "-i", str(replay_file),
        "-o", str(output_dir)
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"🚨 Failed to process {replay_file.name}:\n{result.stderr}")
        return

    # Step 2: Generate CSV with unique name
    try:
        # Load metadata
        with open(output_dir / 'metadata.json', 'r', encoding='utf8') as f:
            metadata = json.load(f)
        
        # Process player data
        player_names = {str(p['unique_id']): p['name'] for p in metadata['players']}
        player_teams = {str(p['unique_id']): (0 if not p['is_orange'] else 1) for p in metadata['players']}

        # Load game data
        game_df = pd.read_parquet(output_dir / '__game.parquet')
        ball_df = pd.read_parquet(output_dir / '__ball.parquet').add_prefix('ball_')

        # Process player files
        all_players = []
        for player_file in output_dir.glob("player_*.parquet"):
            player_id = player_file.stem.split("_")[1]
            player_df = pd.read_parquet(player_file)
            player_df['player_name'] = player_names.get(player_id, "Unknown")
            player_df['team'] = player_teams.get(player_id, -1)
            all_players.append(player_df)
        
        players_df = pd.concat(all_players, ignore_index=True)

        # Align data lengths
        game_df_repeated = pd.concat([game_df] * (len(players_df) // len(game_df)), ignore_index=True).iloc[:len(players_df)]
        ball_df_repeated = pd.concat([ball_df] * (len(players_df) // len(ball_df)), ignore_index=True).iloc[:len(players_df)]
        
        # Merge data
        final_df = pd.concat([game_df_repeated, players_df, ball_df_repeated], axis=1)
        final_df.sort_values(by=['time', 'player_name'], inplace=True)

        # Save CSV with unique name
        csv_name = f"game_positions_{replay_file.stem}.csv"
        final_csv = output_dir / csv_name
        final_df.to_csv(final_csv, index=False)
        print(f"✅ Success: {replay_file.name} → {final_csv}")

        # Step 3: Cleanup
        cleanup_intermediate_files(output_dir)
        
    except Exception as e:
        print(f"❌ Error processing {replay_file.name}: {str(e)}")

def cleanup_intermediate_files(output_dir: Path):
    """Cleans up intermediate files while keeping the final CSV"""
    targets = [
        "analyzer.json",
        "metadata.json",
        "__ball.parquet",
        "__game.parquet",
        "player_*.parquet"
    ]
    
    cleaned = []
    for pattern in targets:
        for file in output_dir.glob(pattern):
            os.remove(file)
            cleaned.append(file.name)
    
    print(f"🧹 Cleaned: {', '.join(cleaned)}")

if __name__ == "__main__":
    start_time = time.time()
    replays = find_replay_files(PARENT_DIR)
    print(f"🔍 Found {len(replays)} replay files")
    
    for idx, replay in enumerate(replays, 1):
        print(f"\n📁 Processing {idx}/{len(replays)}: {replay.name}")
        process_replay(replay)
    
    # Calculate total time
    total_sec = time.time() - start_time
    mins, secs = divmod(total_sec, 60)
    print(f"\n⏱ Total processing time: {int(mins)}m {secs:.1f}s")