import pandas as pd
import json
from pathlib import Path

# Path to the processed replay folder
replay_dir = Path("C:\\Users\\serda\\Downloads")

# Load the metadata
with open(replay_dir / 'metadata.json', 'r', encoding='utf8') as f:
    metadata = json.load(f)

# Extract player names and team numbers using unique_id
player_names = {str(player['unique_id']): player['name'] for player in metadata['players']}
player_teams = {str(player['unique_id']): (0 if not player['is_orange'] else 1) for player in metadata['players']}

# Load the game time data
game_df = pd.read_parquet(replay_dir / '__game.parquet')

# Load the ball data and add `ball_` prefix to its column names
ball_df = pd.read_parquet(replay_dir / '__ball.parquet')
ball_df = ball_df.rename(columns=lambda col: f'ball_{col}')

# Prepare a list to hold all position data
all_positions = []

# Load the player data (parquet file for each player)
for player_file in replay_dir.glob("player_*.parquet"):
    player_id = player_file.stem.split("_")[1]
    player_name = player_names.get(player_id, "Unknown")  # Get player name
    player_team = player_teams.get(player_id, -1)  # Get player team
    
    # Load player data
    player_df = pd.read_parquet(player_file)
    
    # Add columns for player name and team to the dataframe
    player_df['player_name'] = player_name
    player_df['team'] = player_team
    
    # Append this player's data to the all_positions list
    all_positions.append(player_df)

# Concatenate all player data into one DataFrame
players_df = pd.concat(all_positions, ignore_index=True)

# Adjust the length of game and ball data to match players' data
game_df_repeated = pd.concat([game_df] * (len(players_df) // len(game_df)), ignore_index=True)
ball_df_repeated = pd.concat([ball_df] * (len(players_df) // len(ball_df)), ignore_index=True)

# Ensure the length matches exactly (in case of rounding differences)
game_df_repeated = game_df_repeated.iloc[:len(players_df)].reset_index(drop=True)
ball_df_repeated = ball_df_repeated.iloc[:len(players_df)].reset_index(drop=True)

# Merge game time data with player data
merged_df = pd.concat([game_df_repeated, players_df.reset_index(drop=True)], axis=1)

# Merge the result with ball data
final_df = pd.concat([merged_df, ball_df_repeated], axis=1)

# Sort the final DataFrame by time
final_df = final_df.sort_values(by=['time', 'player_name'])

# Print the DataFrame
print(final_df)

# Optionally, you can save it to a CSV file
final_df.to_csv(replay_dir / 'game_positions_with_ball.csv', index=False)
