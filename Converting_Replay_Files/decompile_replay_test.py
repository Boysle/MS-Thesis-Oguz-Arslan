import pandas as pd
from pathlib import Path
from carball.analysis.analysis_manager import AnalysisManager
from carball.decompile_replays import analyze_replay_file

# ==================================================================
# Configuration (Update these paths!)
# ==================================================================
REPLAY_DIR = Path(r"D:\\RL Esports Replays\\Swiss\\Round 1\\AM vs GG")  # Directory containing .replay files
OUTPUT_DIR = Path(r"D:\\RL Esports Replays\\Swiss\\Round 1\\AM vs GG")   # Directory to save CSV files
# ==================================================================

def analyze_demos(replay_path: str, output_dir: Path):
    """
    Analyzes a replay file and saves demo-related information to a CSV file.
    """
    try:
        print(f"📁 Processing: {Path(replay_path).name}")

        # Step 1: Analyze the replay
        analysis = analyze_replay_file(replay_path)

        # Step 2: Get the Protobuf data
        proto_game = analysis.get_protobuf_data()

        # Step 3: Access the bumps (demos) list
        bumps = proto_game.game_stats.bumps

        # Step 4: Create a DataFrame for demos
        demo_data = []
        for bump in bumps:
            demo_data.append({
                "Attacker": bump.attacker.name,
                "Victim": bump.victim.name,
                "Time (seconds)": bump.time_seconds,
                "Is Demo": bump.is_demo
            })
        df = pd.DataFrame(demo_data)

        # Step 5: Save the DataFrame to a CSV file
        csv_path = output_dir / f"{Path(replay_path).stem}_demos.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ Demo data saved: {csv_path}")

    except ParseError as e:
        # Ignore the specific error about "CurrentVoiceRoom"
        if "CurrentVoiceRoom" in str(e):
            print(f"⚠️ Skipping {Path(replay_path).name}: Unsupported attribute (CurrentVoiceRoom).")
        else:
            print(f"❌ Error processing {Path(replay_path).name}: {str(e)}")
    except Exception as e:
        print(f"❌ Unexpected error processing {Path(replay_path).name}: {str(e)}")

def main():
    # Ensure the output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all replay files in the directory
    replay_files = list(REPLAY_DIR.glob("*.replay"))
    print(f"🔍 Found {len(replay_files)} replay files")

    # Process each replay file
    for replay_file in replay_files:
        analyze_demos(str(replay_file), OUTPUT_DIR)

if __name__ == "__main__":
    main()