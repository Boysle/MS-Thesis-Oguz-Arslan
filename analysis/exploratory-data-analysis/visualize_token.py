import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
from collections import defaultdict
import matplotlib.lines as mlines
import os
import seaborn as sns # Seaborn is used for its styling, which matplotlib respects

# ====================== CONFIGURATION & CONSTANTS ======================
POS_MIN_X, POS_MAX_X = -4096, 4096
POS_MIN_Y, POS_MAX_Y = -6000, 6000
POS_MIN_Z, POS_MAX_Z = 0, 2044
GOAL_LINE_Y = 5120
GOAL_WIDTH = 892.755 * 2
GOAL_HEIGHT = 642.775

# ====================== ARGUMENT PARSER ======================
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Visualize a game state token in 3D space.")
    # Token is now optional to allow for interactive mode
    parser.add_argument('--token', type=str, default=None, help='The full token string to visualize.')
    parser.add_argument('--parcel-size', type=int, default=512, help='The size of the grid parcels used to create the token.')
    parser.add_argument('--exclude-player-z', action='store_true', help='Specify if player Z-axis was excluded from the token.')
    parser.add_argument('--exclude-ball-z', action='store_true', help='Specify if ball Z-axis was excluded from the token.')
    parser.add_argument('--output-dir', type=str, default='./token_visualizations', help='Directory to save output plots.')
    return parser.parse_args()

# ====================== CORE FUNCTIONS ======================
def get_parcel_center(px, py, pz, parcel_size):
    """Calculates the real-world 3D coordinate of the center of a parcel."""
    x = (px * parcel_size) + (parcel_size / 2)
    y = (py * parcel_size) + (parcel_size / 2)
    z = (pz * parcel_size) + (parcel_size / 2)
    return x, y, z

def parse_token(token, args):
    """Parses a token string and returns the parcel indices of all objects."""
    parts = re.split(r'_(P_BLUE|P_ORANGE)_', token)
    ball_part = parts[0].split('_')[1:]; blue_part = parts[2].split('_'); orange_part = parts[4].split('_')
    ball_indices = []; blue_player_indices = []; orange_player_indices = []
    
    if args.exclude_ball_z:
        px, py = map(int, ball_part); ball_indices = (px, py, 0)
    else:
        px, py, pz = map(int, ball_part); ball_indices = (px, py, pz)

    step = 2 if args.exclude_player_z else 3
    for i in range(0, len(blue_part), step):
        coords = list(map(int, blue_part[i:i+step])); px, py = coords[0], coords[1]; pz = coords[2] if not args.exclude_player_z else 0
        blue_player_indices.append((px, py, pz))
    for i in range(0, len(orange_part), step):
        coords = list(map(int, orange_part[i:i+step])); px, py = coords[0], coords[1]; pz = coords[2] if not args.exclude_player_z else 0
        orange_player_indices.append((px, py, pz))
            
    return ball_indices, blue_player_indices, orange_player_indices

def plot_field_with_grid(ax, parcel_size):
    """Draws a wireframe of the field, the internal parcel grid, and the goal lines."""
    x_ticks = np.arange(0, POS_MAX_X + 1, parcel_size); x_ticks = np.union1d(x_ticks, -x_ticks)
    y_ticks = np.arange(0, POS_MAX_Y + 1, parcel_size); y_ticks = np.union1d(y_ticks, -y_ticks)
    z_ticks = np.arange(POS_MIN_Z, POS_MAX_Z + 1, parcel_size)

    for z_level in [POS_MIN_Z, POS_MAX_Z]:
        for x in x_ticks: ax.plot([x, x], [POS_MIN_Y, POS_MAX_Y], z_level, color='gray', linestyle=':', linewidth=0.6, alpha=0.5)
        for y in y_ticks: ax.plot([POS_MIN_X, POS_MAX_X], [y, y], z_level, color='gray', linestyle=':', linewidth=0.6, alpha=0.5)
    for x_level in [POS_MIN_X, POS_MAX_X]:
        for y in y_ticks: ax.plot([x_level, x_level], [y, y], [POS_MIN_Z, POS_MAX_Z], color='gray', linestyle=':', linewidth=0.6, alpha=0.5)
        for z in z_ticks: ax.plot([x_level, x_level], [POS_MIN_Y, POS_MAX_Y], z, color='gray', linestyle=':', linewidth=0.6, alpha=0.5)
        
    goal_x_coords = [-GOAL_WIDTH / 2, GOAL_WIDTH / 2]
    ax.plot(goal_x_coords, [-GOAL_LINE_Y, -GOAL_LINE_Y], [GOAL_HEIGHT, GOAL_HEIGHT], color='blue', linewidth=3)
    ax.plot([goal_x_coords[0], goal_x_coords[0]], [-GOAL_LINE_Y, -GOAL_LINE_Y], [POS_MIN_Z, GOAL_HEIGHT], color='blue', linewidth=3)
    ax.plot([goal_x_coords[1], goal_x_coords[1]], [-GOAL_LINE_Y, -GOAL_LINE_Y], [POS_MIN_Z, GOAL_HEIGHT], color='blue', linewidth=3)
    ax.plot(goal_x_coords, [GOAL_LINE_Y, GOAL_LINE_Y], [GOAL_HEIGHT, GOAL_HEIGHT], color='orange', linewidth=3)
    ax.plot([goal_x_coords[0], goal_x_coords[0]], [GOAL_LINE_Y, GOAL_LINE_Y], [POS_MIN_Z, GOAL_HEIGHT], color='orange', linewidth=3)
    ax.plot([goal_x_coords[1], goal_x_coords[1]], [GOAL_LINE_Y, GOAL_LINE_Y], [POS_MIN_Z, GOAL_HEIGHT], color='orange', linewidth=3)

def run_visualization(token_str, args):
    """The core logic for parsing and plotting a single token."""
    print(f"\n--- Visualizing Token ---")
    try:
        ball_indices, blue_indices, orange_indices = parse_token(token_str, args)
    except (ValueError, IndexError) as e:
        print(f"\nCRITICAL ERROR: Could not parse the token. Please check the format.")
        print(f"Details: {e}")
        return

    parcel_occupants = defaultdict(list)
    parcel_occupants[ball_indices].append({'type': 'ball', 'color': 'darkgreen', 'size': 200})
    for p_idx in blue_indices: parcel_occupants[p_idx].append({'type': 'player', 'color': 'blue', 'size': 120})
    for p_idx in orange_indices: parcel_occupants[p_idx].append({'type': 'player', 'color': 'orange', 'size': 120})

    # Apply seaborn styling for a professional look
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(12, 14)); ax = fig.add_subplot(111, projection='3d')
    plot_field_with_grid(ax, args.parcel_size)
    
    for (px, py, pz), occupants in parcel_occupants.items():
        base_x, base_y, base_z = get_parcel_center(px, py, pz, args.parcel_size)
        if len(occupants) == 1:
            obj = occupants[0]; ax.scatter(base_x, base_y, base_z, c=obj['color'], s=obj['size'], depthshade=True)
        else:
            jitter_strength = args.parcel_size * 0.25
            for obj in occupants:
                jitter_x = base_x + (np.random.rand() - 0.5) * jitter_strength
                jitter_y = base_y + (np.random.rand() - 0.5) * jitter_strength
                jitter_z = base_z + (np.random.rand() - 0.5) * jitter_strength
                ax.scatter(jitter_x, jitter_y, jitter_z, c=obj['color'], s=obj['size'], depthshade=True)
            ax.text(base_x, base_y, base_z, f'  x{len(occupants)}', color='red', fontsize=12, fontweight='bold')

    x_ticks = np.arange(0, POS_MAX_X + 1, args.parcel_size); x_ticks = np.union1d(x_ticks, -x_ticks)
    y_ticks = np.arange(0, POS_MAX_Y + 1, args.parcel_size); y_ticks = np.union1d(y_ticks, -y_ticks)
    z_ticks = np.arange(POS_MIN_Z, POS_MAX_Z + 1, args.parcel_size)
    ax.set_xticks(x_ticks); ax.set_yticks(y_ticks); ax.set_zticks(z_ticks)
    
    ax.set_xlabel('X Axis'); ax.set_ylabel('Y Axis (Orange Goal ->)'); ax.set_zlabel('Z Axis')
    ax.set_xlim(POS_MIN_X, POS_MAX_X); ax.set_ylim(POS_MIN_Y, POS_MAX_Y); ax.set_zlim(POS_MIN_Z, POS_MAX_Z)
    ax.set_box_aspect(((POS_MAX_X - POS_MIN_X), (POS_MAX_Y - POS_MIN_Y), (POS_MAX_Z - POS_MIN_Z)))
    plt.title('3D Visualization of Game State Token', fontsize=16)

    legend_elements = [mlines.Line2D([0], [0], marker='o', color='w', label='Ball', markerfacecolor='darkgreen', markersize=10),
                       mlines.Line2D([0], [0], marker='o', color='w', label='Blue Team', markerfacecolor='blue', markersize=10),
                       mlines.Line2D([0], [0], marker='o', color='w', label='Orange Team', markerfacecolor='orange', markersize=10)]
    ax.legend(handles=legend_elements)
    
    # Create a unique filename for each token visualization
    token_hash = hex(hash(token_str) & 0xffffffff)[2:] # Simple hash for unique filename
    output_filename = f'token_viz_{token_hash}.png'
    filepath = os.path.join(args.output_dir, output_filename)
    os.makedirs(args.output_dir, exist_ok=True)

    plt.savefig(filepath, dpi=150)
    print(f"\nVisualization saved to {filepath}")
    plt.show()

# ====================== MAIN EXECUTION ======================
def main():
    args = parse_args()
    
    if args.token:
        # If a token is provided via command line, run once and exit.
        print("--- Running in single-run mode ---")
        run_visualization(args.token, args)
    else:
        # If no token is provided, enter interactive mode.
        print("\n--- Welcome to the Interactive Token Visualizer ---")
        print("This program uses the following settings:")
        print(f"  - Parcel Size: {args.parcel_size}")
        print(f"  - Exclude Player Z: {'Yes' if args.exclude_player_z else 'No'}")
        print(f"  - Exclude Ball Z:   {'Yes' if args.exclude_ball_z else 'No'}")
        print("\nPaste a token and press Enter to generate a plot.")
        print("Type 'exit' or 'quit' to close the program.")
        
        while True:
            try:
                token_input = input("\nEnter a token > ")
                if token_input.lower() in ['exit', 'quit']:
                    print("Exiting program. Goodbye!"); break
                if not token_input.strip(): continue
                run_visualization(token_input.strip(), args)
            except KeyboardInterrupt:
                print("\nExiting program. Goodbye!"); break
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()