import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
from collections import defaultdict
import matplotlib.lines as mlines
import os
import seaborn as sns

# ====================== CONFIGURATION & CONSTANTS ======================
POS_MIN_X, POS_MAX_X = -4096, 4096
POS_MIN_Y, POS_MAX_Y = -6000, 6000
POS_MIN_Z, POS_MAX_Z = 0, 2044
GOAL_LINE_Y = 5120
GOAL_WIDTH = 893 * 2
GOAL_HEIGHT = 642.775
CORNER_CUTOFF_X = 2944 # 4096 - 1152
CORNER_CUTOFF_Y = 3968 # 5120 - 1152

# ====================== CORE FUNCTIONS ======================
def parse_token(token_str, parcel_size, exclude_player_z, exclude_ball_z):
    # This function is correct and remains unchanged from the previous version.
    parts = re.split(r'_(P_BLUE|P_ORANGE)_', token_str)
    ball_part = parts[0].split('_')[1:]; blue_part = parts[2].split('_'); orange_part = parts[4].split('_')
    ball_indices = []; blue_player_indices = []; orange_player_indices = []
    
    if exclude_ball_z:
        px, py = map(int, ball_part); ball_indices = (px, py, 0)
    else:
        px, py, pz = map(int, ball_part); ball_indices = (px, py, pz)

    step = 2 if exclude_player_z else 3
    for i in range(0, len(blue_part), step):
        coords = list(map(int, blue_part[i:i+step])); px, py = coords[0], coords[1]; pz = coords[2] if not exclude_player_z else 0
        blue_player_indices.append((px, py, pz))
    for i in range(0, len(orange_part), step):
        coords = list(map(int, orange_part[i:i+step])); px, py = coords[0], coords[1]; pz = coords[2] if not exclude_player_z else 0
        orange_player_indices.append((px, py, pz))
            
    return ball_indices, blue_player_indices, orange_player_indices

def get_parcel_center(px, py, pz, parcel_size):
    # This function is correct and remains unchanged.
    x = (px * parcel_size) + (parcel_size / 2); y = (py * parcel_size) + (parcel_size / 2); z = (pz * parcel_size) + (parcel_size / 2)
    return x, y, z

def plot_field_with_grid(ax, parcel_size):
    """Draws the accurate field shape, internal grid, and goal lines."""
    # --- NEW: Accurate Field Shape Vertices ---
    field_vertices = [
        (CORNER_CUTOFF_X, GOAL_LINE_Y), (GOAL_WIDTH/2, GOAL_LINE_Y), (-GOAL_WIDTH/2, GOAL_LINE_Y), (-CORNER_CUTOFF_X, GOAL_LINE_Y),
        (-POS_MAX_X, CORNER_CUTOFF_Y), (-POS_MAX_X, -CORNER_CUTOFF_Y),
        (-CORNER_CUTOFF_X, -GOAL_LINE_Y), (-GOAL_WIDTH/2, -GOAL_LINE_Y), (GOAL_WIDTH/2, -GOAL_LINE_Y), (CORNER_CUTOFF_X, -GOAL_LINE_Y),
        (POS_MAX_X, -CORNER_CUTOFF_Y), (POS_MAX_X, CORNER_CUTOFF_Y), (CORNER_CUTOFF_X, GOAL_LINE_Y) # Close the loop
    ]
    field_x, field_y = zip(*field_vertices)

    # Draw floor and ceiling outlines
    ax.plot(field_x, field_y, zs=0, zdir='z', color='black', linewidth=1.5)
    ax.plot(field_x, field_y, zs=POS_MAX_Z, zdir='z', color='black', linewidth=1)

    # Draw vertical corner posts
    for vx, vy in field_vertices[:-1]: # Exclude the last point to avoid double-drawing
        ax.plot([vx, vx], [vy, vy], [0, POS_MAX_Z], color='black', linewidth=1)
    
    # --- Internal Grid (remains the same) ---
    x_ticks = np.arange(0, POS_MAX_X + 1, parcel_size); x_ticks = np.union1d(x_ticks, -x_ticks)
    y_ticks = np.arange(0, POS_MAX_Y + 1, parcel_size); y_ticks = np.union1d(y_ticks, -y_ticks)
    for z_level in [POS_MIN_Z, POS_MAX_Z]:
        for x in x_ticks: ax.plot([x, x], [POS_MIN_Y, POS_MAX_Y], z_level, color='gray', linestyle=':', linewidth=0.6, alpha=0.5)
        for y in y_ticks: ax.plot([POS_MIN_X, POS_MAX_X], [y, y], z_level, color='gray', linestyle=':', linewidth=0.6, alpha=0.5)

    # --- Goal Lines (remains the same) ---
    goal_x_coords = [-GOAL_WIDTH / 2, GOAL_WIDTH / 2]
    ax.plot(goal_x_coords, [-GOAL_LINE_Y, -GOAL_LINE_Y], [GOAL_HEIGHT, GOAL_HEIGHT], color='blue', linewidth=3) # Blue Crossbar
    ax.plot([goal_x_coords[0], goal_x_coords[0]], [-GOAL_LINE_Y, -GOAL_LINE_Y], [0, GOAL_HEIGHT], color='blue', linewidth=3)
    ax.plot([goal_x_coords[1], goal_x_coords[1]], [-GOAL_LINE_Y, -GOAL_LINE_Y], [0, GOAL_HEIGHT], color='blue', linewidth=3)
    ax.plot(goal_x_coords, [GOAL_LINE_Y, GOAL_LINE_Y], [GOAL_HEIGHT, GOAL_HEIGHT], color='orange', linewidth=3) # Orange Crossbar
    ax.plot([goal_x_coords[0], goal_x_coords[0]], [GOAL_LINE_Y, GOAL_LINE_Y], [0, GOAL_HEIGHT], color='orange', linewidth=3)
    ax.plot([goal_x_coords[1], goal_x_coords[1]], [GOAL_LINE_Y, GOAL_LINE_Y], [0, GOAL_HEIGHT], color='orange', linewidth=3)

def generate_static_views(fig, ax, output_dir, base_filename):
    """Saves the plot from canonical top, side, and front views."""
    views = {
        'top': {'elev': 90, 'azim': -90},
        'side': {'elev': 0, 'azim': 0},
        'front': {'elev': 0, 'azim': -90}
    }
    filepaths = {}
    for view_name, angles in views.items():
        ax.view_init(elev=angles['elev'], azim=angles['azim'])
        filepath = os.path.join(output_dir, f"{base_filename}_{view_name}_view.png")
        plt.savefig(filepath, dpi=120)
        filepaths[view_name] = filepath
    return filepaths

def create_visualization_figure(token_str, parcel_size, exclude_player_z, exclude_ball_z, output_dir):
    """The core logic for parsing and plotting, returning the main figure and static paths."""
    try:
        ball_indices, blue_indices, orange_indices = parse_token(token_str, parcel_size, exclude_player_z, exclude_ball_z)
    except Exception as e:
        st.error(f"Could not parse the token. Please check format.\nDetails: {e}")
        return None, None

    # ... (Jittering logic remains the same)
    parcel_occupants = defaultdict(list)
    parcel_occupants[ball_indices].append({'type': 'ball', 'color': 'lightgrey', 'size': 200, 'edgecolor': 'black'})
    for p_idx in blue_indices: parcel_occupants[p_idx].append({'type': 'player', 'color': 'blue', 'size': 120, 'edgecolor': 'white'})
    for p_idx in orange_indices: parcel_occupants[p_idx].append({'type': 'player', 'color': 'orange', 'size': 120, 'edgecolor': 'white'})

    sns.set_theme(style="whitegrid"); fig = plt.figure(figsize=(12, 14)); ax = fig.add_subplot(111, projection='3d')
    plot_field_with_grid(ax, parcel_size)
    
    for (px, py, pz), occupants in parcel_occupants.items():
        base_x, base_y, base_z = get_parcel_center(px, py, pz, parcel_size)
        if len(occupants) == 1:
            obj = occupants[0]
            ax.scatter(base_x, base_y, base_z, c=obj['color'], s=obj['size'], edgecolors=obj['edgecolor'], depthshade=True)
        else:
            jitter_strength = parcel_size * 0.35 # Increased jitter
            for obj in occupants:
                jitter_x = base_x + (np.random.rand() - 0.5) * jitter_strength
                jitter_y = base_y + (np.random.rand() - 0.5) * jitter_strength
                jitter_z = base_z + (np.random.rand() - 0.5) * jitter_strength
                ax.scatter(jitter_x, jitter_y, jitter_z, c=obj['color'], s=obj['size'], edgecolors=obj['edgecolor'], depthshade=True)
            ax.text(base_x, base_y, base_z, f'  x{len(occupants)}', color='red', fontsize=12, fontweight='bold')

    x_ticks = np.arange(0, POS_MAX_X + 1, parcel_size); x_ticks = np.union1d(x_ticks, -x_ticks)
    y_ticks = np.arange(0, POS_MAX_Y + 1, parcel_size); y_ticks = np.union1d(y_ticks, -y_ticks)
    z_ticks = np.arange(POS_MIN_Z, POS_MAX_Z + 1, parcel_size)
    ax.set_xticks(x_ticks); ax.set_yticks(y_ticks); ax.set_zticks(z_ticks)
    
    ax.set_xlabel('X Axis'); ax.set_ylabel('Y Axis'); ax.set_zlabel('Z Axis')
    ax.set_xlim(POS_MIN_X, POS_MAX_X); ax.set_ylim(POS_MIN_Y, POS_MAX_Y); ax.set_zlim(POS_MIN_Z, POS_MAX_Z)
    ax.set_box_aspect(((POS_MAX_X - POS_MIN_X), (POS_MAX_Y - POS_MIN_Y), (POS_MAX_Z - POS_MIN_Z)))
    plt.title('3D Visualization of Game State Token', fontsize=16)

    legend_elements = [mlines.Line2D([0], [0], marker='o', color='w', label='Ball', markerfacecolor='lightgrey', markeredgecolor='black', markersize=10),
                       mlines.Line2D([0], [0], marker='o', color='w', label='Blue Team', markerfacecolor='blue', markersize=10),
                       mlines.Line2D([0], [0], marker='o', color='w', label='Orange Team', markerfacecolor='orange', markersize=10)]
    ax.legend(handles=legend_elements)

    # --- NEW: Generate static views BEFORE returning the main figure ---
    token_hash = hex(hash(token_str) & 0xffffffff)[2:]
    base_filename = f'token_viz_{token_hash}'
    static_paths = generate_static_views(fig, ax, output_dir, base_filename)
    
    # Reset to a nice isometric view for the interactive plot
    ax.view_init(elev=30, azim=-60)
    
    return fig, static_paths

# ====================== STREAMLIT APP ======================
st.set_page_config(layout="wide")
st.title("🚀 Rocket League Statistical Token Visualizer")

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    token_input = st.text_area("Paste Token String Here:", "B_-3_4_0_P_BLUE_-4_2_0_-3_4_0_-1_2_0_P_ORANGE_-3_4_0_-2_4_0_-1_4_0", height=150)
    parcel_size = st.number_input("Parcel Size", min_value=128, max_value=2048, value=1024, step=128)
    st.write("---"); exclude_player_z = st.checkbox("Exclude Player Z-Axis"); exclude_ball_z = st.checkbox("Exclude Ball Z-Axis"); st.write("---")
    generate_button = st.button("Generate Visualization", type="primary")

# --- Main panel ---
if generate_button:
    if token_input.strip():
        output_dir = "./token_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        with st.spinner('Generating 3D plot and static views...'):
            fig, static_paths = create_visualization_figure(
                token_str=token_input.strip(), parcel_size=parcel_size,
                exclude_player_z=exclude_player_z, exclude_ball_z=exclude_ball_z,
                output_dir=output_dir
            )
            if fig and static_paths:
                st.header("Interactive 3D View")
                st.pyplot(fig)
                
                st.header("Static Views")

                st.subheader("Top-Down View")
                st.image(static_paths['top'])

                st.subheader("Side View")
                st.image(static_paths['side'])

                st.subheader("Front View")
                st.image(static_paths['front'])
    else:
        st.warning("Please paste a token string in the sidebar.")
else:
    st.info("⬅️ Enter a token and set the configuration, then click 'Generate Visualization'.")