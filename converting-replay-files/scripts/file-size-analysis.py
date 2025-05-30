import pandas as pd
from pathlib import Path
import os
from collections import defaultdict

def format_size(size_bytes):
    """Convert bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def is_match_folder(path: Path):
    """A folder is considered a match if it contains .replay files
    and is not an Output folder"""
    if path.name.startswith("Output-"):
        return False
    return any(path.glob("*.replay"))

def analyze_filesystem(parent_dir: Path):
    stats = {
        'replay_sizes': [],
        'csv_sizes': [],
        'matches': defaultdict(dict),
        'total_matches': 0,
        'total_games': 0
    }

    for root, dirs, files in os.walk(parent_dir):
        current_path = Path(root)
        
        # Skip processing of Output directories
        dirs[:] = [d for d in dirs if not d.startswith("Output-")]
        
        if not is_match_folder(current_path):
            continue

        # Process match folder
        replays = list(current_path.glob("*.replay"))
        if not replays:
            continue
            
        match_name = current_path.name
        stats['total_matches'] += 1
        game_count = len(replays)
        stats['total_games'] += game_count
        
        # Store match info
        stats['matches'][match_name] = {
            'game_count': game_count,
            'replays': [],
            'csvs': []
        }
        
        # Process each game
        for replay in replays:
            # Record replay size
            replay_size = replay.stat().st_size
            stats['replay_sizes'].append(replay_size)
            stats['matches'][match_name]['replays'].append(replay_size)
            
            # Find corresponding CSV
            output_dir = current_path / f"Output-{replay.stem}"
            csv_file = output_dir / f"game_positions_{replay.stem}.csv"
            
            if csv_file.exists():
                csv_size = csv_file.stat().st_size
                stats['csv_sizes'].append(csv_size)
                stats['matches'][match_name]['csvs'].append(csv_size)

    return stats

def print_statistics(stats):
    print("\n📊 FILE SIZE STATISTICS")
    print("=" * 55)
    
    # Replay files stats
    print("\n🔷 Replay Files:")
    if stats['replay_sizes']:
        s = pd.Series(stats['replay_sizes'])
        print(f"  Total: {format_size(s.sum())}")
        print(f"  Average: {format_size(s.mean())}")
        print(f"  Files: {len(s):,}")
    else:
        print("  No replay files found")
    
    # CSV files stats
    print("\n🔷 CSV Files:")
    if stats['csv_sizes']:
        s = pd.Series(stats['csv_sizes'])
        print(f"  Total: {format_size(s.sum())}")
        print(f"  Average: {format_size(s.mean())}")
        print(f"  Files: {len(s):,}")
    else:
        print("  No CSV files found")

    # Match statistics
    print("\n\n🎮 MATCH STATISTICS")
    print("=" * 55)
    print(f"Total Matches Found: {stats['total_matches']:,}")
    print(f"Total Games Found: {stats['total_games']:,}")
    
    if stats['total_matches'] > 0:
        game_counts = [m['game_count'] for m in stats['matches'].values()]
        s = pd.Series(game_counts)
        
        print("\n📈 Games Per Match:")
        print(f"  Average: {s.mean():.2f}")
        print(f"  Median: {s.median():.2f}")
        print(f"  Min: {s.min()}")
        print(f"  Max: {s.max()}")
        
        # Show match examples
        print("\n🏆 Sample Matches:")
        for match_name, data in list(stats['matches'].items())[:3]:
            print(f"  ▸ {match_name}: {data['game_count']} games")
            if data['csvs']:
                avg_size = format_size(sum(data['csvs'])/len(data['csvs']))
                print(f"    Avg CSV size: {avg_size}")

def main():
    # ==================================================================
    # Configuration
    # ==================================================================
    PARENT_DIR = Path(r"E:\RL Esports Replays")  # Update this path
    # ==================================================================
    
    print("🔍 Scanning directory structure...")
    stats = analyze_filesystem(PARENT_DIR)
    
    print("\n" + "="*55)
    print("📈 ANALYSIS REPORT".center(55))
    print("="*55)
    
    print_statistics(stats)
    print("\n" + "="*55)
    print("Analysis complete! 🚀".center(55))
    print("="*55)

if __name__ == "__main__":
    main()