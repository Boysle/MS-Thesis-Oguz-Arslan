import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
# import seaborn as sns # Not strictly needed if using pure matplotlib for plots

# --- Configuration (MUST MATCH YOUR TRAINING SCRIPT) ---
# These should ideally be loaded from a shared config file or hardcoded
# but MUST be identical to what was used during data processing for the model.
NUM_PLAYERS = 6
PLAYER_FEATURES_COUNT = 13 
GLOBAL_FEATURE_DIM_COUNT = 9

# Normalization bounds (critical to be the same)
POS_MIN_X, POS_MAX_X = -4096, 4096
POS_MIN_Y, POS_MAX_Y = -6000, 6000
POS_MIN_Z, POS_MAX_Z = 0, 2044
VEL_MIN, VEL_MAX = -2300, 2300
BOOST_MIN, BOOST_MAX = 0, 100
BALL_VEL_MIN, BALL_VEL_MAX = -6000, 6000
BOOST_PAD_MIN, BOOST_PAD_MAX = 0, 10
DIST_MIN, DIST_MAX = 0, (8192**2 + 10240**2 + 2044**2)**(1/2)

# Feature Names for plotting (ensure order matches your data processing)
PLAYER_FEATURE_NAMES_UI = [
    'p_pos_x', 'p_pos_y', 'p_pos_z',
    'p_vel_x', 'p_vel_y', 'p_vel_z',
    'p_forward_x', 'p_forward_y', 'p_forward_z',
    'p_boost_amount', 'p_team', 'p_alive', 'p_dist_to_ball'
]
GLOBAL_FEATURE_NAMES_UI = [
    'ball_pos_x', 'ball_pos_y', 'ball_pos_z',
    'ball_vel_x', 'ball_vel_y', 'ball_vel_z',
    'boost_pad_0_respawn', 'ball_hit_team_num', 'seconds_remaining'
]
# --- End Configuration ---

# Initialize session state for the toggle if it doesn't exist
if 'highlight_test_set' not in st.session_state:
    st.session_state.highlight_test_set = True # Default to highlighting ON
if 'selected_timeline_idx' not in st.session_state: # For persisting single state inspection
    st.session_state.selected_timeline_idx = 0
if 'selected_filtered_timeline_idx' not in st.session_state: # For persisting inspection on filtered data
     st.session_state.selected_filtered_timeline_idx = 0


# --- Data Preprocessing Logic (Mirrors training script's load_and_process_data) ---
def normalize_ui(val, min_val, max_val):
    return (val - min_val) / (max_val - min_val + 1e-8)

def reconstruct_features_from_csv_row(csv_row_series):
    player_features_list = []
    for i in range(NUM_PLAYERS):
        pos_x = normalize_ui(float(csv_row_series[f'p{i}_pos_x']), POS_MIN_X, POS_MAX_X)
        pos_y = normalize_ui(float(csv_row_series[f'p{i}_pos_y']), POS_MIN_Y, POS_MAX_Y)
        pos_z = normalize_ui(float(csv_row_series[f'p{i}_pos_z']), POS_MIN_Z, POS_MAX_Z)
        vel_x = normalize_ui(float(csv_row_series[f'p{i}_vel_x']), VEL_MIN, VEL_MAX)
        vel_y = normalize_ui(float(csv_row_series[f'p{i}_vel_y']), VEL_MIN, VEL_MAX)
        vel_z = normalize_ui(float(csv_row_series[f'p{i}_vel_z']), VEL_MIN, VEL_MAX)
        forward_x = float(csv_row_series[f'p{i}_forward_x'])
        forward_y = float(csv_row_series[f'p{i}_forward_y'])
        forward_z = float(csv_row_series[f'p{i}_forward_z'])
        boost = normalize_ui(float(csv_row_series[f'p{i}_boost_amount']), BOOST_MIN, BOOST_MAX)
        team = float(csv_row_series[f'p{i}_team']) 
        alive = float(csv_row_series[f'p{i}_alive']) 
        dist_to_ball = normalize_ui(float(csv_row_series[f'p{i}_dist_to_ball']), DIST_MIN, DIST_MAX)

        player_features_list.append([
            pos_x, pos_y, pos_z, vel_x, vel_y, vel_z,
            forward_x, forward_y, forward_z, boost, team, alive, dist_to_ball
        ])
    player_features_np = np.array(player_features_list, dtype=np.float32)

    seconds_remaining_val = float(csv_row_series['seconds_remaining'])
    normalized_seconds = normalize_ui(min(seconds_remaining_val, 300.0), 0, 300)

    global_features_np = np.array([
        normalize_ui(float(csv_row_series['ball_pos_x']), POS_MIN_X, POS_MAX_X),
        normalize_ui(float(csv_row_series['ball_pos_y']), POS_MIN_Y, POS_MAX_Y),
        normalize_ui(float(csv_row_series['ball_pos_z']), POS_MIN_Z, POS_MAX_Z),
        normalize_ui(float(csv_row_series['ball_vel_x']), BALL_VEL_MIN, BALL_VEL_MAX),
        normalize_ui(float(csv_row_series['ball_vel_y']), BALL_VEL_MIN, BALL_VEL_MAX),
        normalize_ui(float(csv_row_series['ball_vel_z']), BALL_VEL_MIN, BALL_VEL_MAX),
        normalize_ui(float(csv_row_series['boost_pad_0_respawn']), BOOST_PAD_MIN, BOOST_PAD_MAX),
        float(csv_row_series['ball_hit_team_num']), 
        normalized_seconds
    ], dtype=np.float32)

    assert player_features_np.shape == (NUM_PLAYERS, PLAYER_FEATURES_COUNT), "Player features shape mismatch in UI"
    assert global_features_np.shape == (GLOBAL_FEATURE_DIM_COUNT,), "Global features shape mismatch in UI"
    return player_features_np, global_features_np
# --- End Data Preprocessing Logic ---


# --- Plotting Function (Mirrors training script's plot_avg_positions) ---
def plot_avg_positions_ui(player_features_np, global_features_np, plot_title_suffix, team_id_feature_idx):
    avg_player_pos = player_features_np[:, :2] 
    avg_ball_pos = global_features_np[:2]   

    fig, ax = plt.subplots(figsize=(7, 9))
    
    ax.axhline(y=normalize_ui(POS_MAX_Y, POS_MIN_Y, POS_MAX_Y) - 0.02, color='orange', linestyle='--', xmin=0.35, xmax=0.65, lw=2, label="Orange Goal Area")
    ax.axhline(y=normalize_ui(POS_MIN_Y, POS_MIN_Y, POS_MAX_Y) + 0.02, color='blue', linestyle='--', xmin=0.35, xmax=0.65, lw=2, label="Blue Goal Area")
    ax.set_xlim(normalize_ui(POS_MIN_X,POS_MIN_X,POS_MAX_X) - 0.05, normalize_ui(POS_MAX_X,POS_MIN_X,POS_MAX_X) + 0.05)
    ax.set_ylim(normalize_ui(POS_MIN_Y,POS_MIN_Y,POS_MAX_Y) - 0.05, normalize_ui(POS_MAX_Y,POS_MIN_Y,POS_MAX_Y) + 0.05)

    player_colors = {0.0: 'blue', 1.0: 'orange'}
    for i in range(NUM_PLAYERS):
        player_team_id = player_features_np[i, team_id_feature_idx]
        color = player_colors.get(player_team_id, 'gray')
        ax.scatter(avg_player_pos[i, 0], avg_player_pos[i, 1],
                    s=120, label=f'P{i}', color=color, edgecolors='black', alpha=0.8)
                    
    ax.scatter(avg_ball_pos[0], avg_ball_pos[1], s=180, color='darkgreen', marker='o', label='Ball', edgecolors='black', alpha=0.9)

    ax.set_title(f'State: {plot_title_suffix}')
    ax.set_xlabel('X Position (Normalized)')
    ax.set_ylabel('Y Position (Normalized)')
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.7)
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    
    return fig
# --- End Plotting Function ---


# --- Timeline String View Function ---
def get_timeline_strings_html(chrono_data, start_idx, end_idx, highlight_test_active):
    """
    Generates HTML for the timeline view. Each character cell shows its
    timeline_idx on hover. Characters are adjacent.
    """
    if not chrono_data or start_idx >= len(chrono_data) or start_idx < 0:
        return "<p>No data to display for this range.</p>"
    
    subset_data = chrono_data[start_idx:end_idx]

    color_map_orange = {(0, 0): "#28a745", (1, 1): "#007bff", (0, 1): "#dc3545", (1, 0): "#ffc107"}
    color_map_blue = {(0, 0): "#17a2b8", (1, 1): "#6f42c1", (0, 1): "#fd7e14", (1, 0): "#e83e8c"}
    default_text_color = "#333333"

    html_lines = {
        "Orange True": "", "Orange Pred": "",
        "Blue True": "", "Blue Pred": ""
    }

    for i, item in enumerate(subset_data):
        actual_timeline_idx = item['timeline_idx'] 
        ot, op = item['orange_true'], item['orange_pred_label']
        bt, bp = item['blue_true'], item['blue_pred_label']
        split_type = item.get('split', 'unknown')
        is_test = split_type == 'test'
        
        current_highlight_style = "background-color: #e9e9e9;" if is_test and highlight_test_active else ""
        
        tooltip_text = f"Timeline Idx: {actual_timeline_idx}\nOriginal Idx: {item['original_idx']}\nSet: {split_type.capitalize()}"

        def styled_data_cell(val, color, bold=False, title_text=""):
            style_parts = [f"color:{color};"]
            if bold: style_parts.append("font-weight:bold;")
            if current_highlight_style: style_parts.append(current_highlight_style)
            
            # Style for consistent cell appearance, making characters adjacent but distinct
            style_parts.append("display:inline-block; width:1em; text-align:center; padding: 1px 0;") 

            safe_title_text = title_text.replace('"', '"')
            
            # NO SPACE appended here
            return f"<span style='{' '.join(style_parts)}' title='{safe_title_text}'>{val}</span>"

        html_lines["Orange True"] += styled_data_cell(ot, color_map_orange.get((ot, ot), default_text_color), bold=True, title_text=tooltip_text)
        html_lines["Orange Pred"] += styled_data_cell(op, color_map_orange.get((ot, op), default_text_color), title_text=tooltip_text)
        html_lines["Blue True"] += styled_data_cell(bt, color_map_blue.get((bt, bt), default_text_color), bold=True, title_text=tooltip_text)
        html_lines["Blue Pred"] += styled_data_cell(bp, color_map_blue.get((bt, bp), default_text_color), title_text=tooltip_text)

    # No need to rstrip lines if no spaces were added between spans

    final_html_str = f"""
    <div style="font-family: 'Courier New', Courier, monospace; font-size: 14px; white-space: pre; overflow-x: auto; border: 1px solid #ccc; padding: 10px; background-color: #f8f9fa; color: {default_text_color}; line-height: 1.5em;">
Orange True: {html_lines['Orange True']}
Orange Pred: {html_lines['Orange Pred']}
Blue True  : {html_lines['Blue True']}
Blue Pred  : {html_lines['Blue Pred']}
    </div>
    <small><i>Hover over any prediction character to see its Timeline Index, Original Index, and Set type. Highlight: Test Set samples.</i></small>
    """
    return final_html_str

# --- End Timeline String View Function ---


# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("Rocket League GCN Model Analysis")

# File Uploaders
uploaded_json_file = st.sidebar.file_uploader("Upload Chronological JSON Results", type="json")
uploaded_csv_file = st.sidebar.file_uploader("Upload Original Game Data CSV", type="csv")

# Add a button to toggle test set highlighting
if st.sidebar.button("Toggle Test Set Highlighting"):
    st.session_state.highlight_test_set = not st.session_state.highlight_test_set

highlight_status_text = "ON" if st.session_state.highlight_test_set else "OFF"
st.sidebar.caption(f"Test Set Highlighting: {highlight_status_text}")


# Load data only once
@st.cache_data 
def load_json_data(file):
    return json.load(file)

@st.cache_data 
def load_csv_data(file):
    return pd.read_csv(file, low_memory=False) 

chrono_data_full = None # Will hold all data from JSON
original_csv_df = None  # Will hold all data from CSV

if uploaded_json_file:
    chrono_data_full = load_json_data(uploaded_json_file)
    st.sidebar.success(f"Loaded {len(chrono_data_full)} states from JSON.")

if uploaded_csv_file:
    original_csv_df = load_csv_data(uploaded_csv_file)
    st.sidebar.success(f"Loaded CSV with {len(original_csv_df)} rows.")


# Main content area - only proceed if both files are loaded
if chrono_data_full and original_csv_df is not None and not original_csv_df.empty:
    
    # Optional: Filter by dataset split
    filtered_chrono_data = chrono_data_full # Default to all data
    available_splits = sorted(list(set(item.get('split', 'unknown') for item in chrono_data_full)))
    if len(available_splits) > 1: 
        selected_split_filter = st.sidebar.selectbox(
            "Filter by Dataset Split:", 
            options=["All"] + available_splits, 
            index=0,
            key="split_filter_select"
        )
        if selected_split_filter != "All":
            filtered_chrono_data = [item for item in chrono_data_full if item.get('split') == selected_split_filter]
    
    st.sidebar.info(f"Displaying {len(filtered_chrono_data)} of {len(chrono_data_full)} total states based on filter.")


    st.header("Timeline View")
    if not filtered_chrono_data:
        st.warning("No data matches the current filter criteria.")
    else:
        timeline_window_size = st.slider("Timeline Window Size", min_value=10, max_value=200, value=50, step=10, key="timeline_window_slider")
        
        max_start_idx_filtered = len(filtered_chrono_data) - timeline_window_size
        if max_start_idx_filtered < 0: max_start_idx_filtered = 0
            
        timeline_start_idx_filtered = st.slider("Timeline Start Index", min_value=0, max_value=max_start_idx_filtered, value=0, step=max(1, timeline_window_size//2), key="timeline_start_slider")
        timeline_end_idx_filtered = min(timeline_start_idx_filtered + timeline_window_size, len(filtered_chrono_data))
        
        st.markdown(
            get_timeline_strings_html(
                filtered_chrono_data, 
                timeline_start_idx_filtered, 
                timeline_end_idx_filtered,
                st.session_state.highlight_test_set 
            ), 
            unsafe_allow_html=True
        )
        st.caption(f"Displaying states from filtered timeline index {timeline_start_idx_filtered} to {timeline_end_idx_filtered-1}")


        st.header("Inspect Single State")
        
        max_inspect_idx_filtered = len(filtered_chrono_data) - 1
        if max_inspect_idx_filtered < 0: # Handle if filtered_chrono_data is empty
             st.info("No states available for inspection with current filters.")
        else:
            # Persist selected index within the filtered view
            current_default_inspect_idx = st.session_state.get('selected_filtered_timeline_idx', 0)
            if current_default_inspect_idx > max_inspect_idx_filtered: # Ensure default is within bounds
                current_default_inspect_idx = 0
            
            inspect_filtered_timeline_idx = st.number_input(
                f"Enter Index from Current Timeline to Inspect (0 to {max_inspect_idx_filtered})", 
                min_value=0, max_value=max_inspect_idx_filtered, 
                value=current_default_inspect_idx, 
                step=1, key="inspect_idx_input"
            )
            st.session_state.selected_filtered_timeline_idx = inspect_filtered_timeline_idx


            if 0 <= inspect_filtered_timeline_idx < len(filtered_chrono_data):
                selected_state_info = filtered_chrono_data[inspect_filtered_timeline_idx] 
                original_data_idx = selected_state_info['original_idx'] 
                split_type_display = selected_state_info.get('split', 'unknown').capitalize()

                st.subheader(f"State Details (Timeline Index: {selected_state_info['timeline_idx']}, Original CSV Idx: {original_data_idx}, Set: {split_type_display})")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Orange Team:**")
                    st.write(f"  True Goal: `{selected_state_info['orange_true']}`")
                    st.write(f"  Predicted Goal: `{selected_state_info['orange_pred_label']}` (Prob: `{selected_state_info['orange_pred_prob']:.3f}`)")
                with col2:
                    st.write(f"**Blue Team:**")
                    st.write(f"  True Goal: `{selected_state_info['blue_true']}`")
                    st.write(f"  Predicted Goal: `{selected_state_info['blue_pred_label']}` (Prob: `{selected_state_info['blue_pred_prob']:.3f}`)")

                if 0 <= original_data_idx < len(original_csv_df):
                    csv_row = original_csv_df.iloc[original_data_idx]
                    try:
                        player_feat_np, global_feat_np = reconstruct_features_from_csv_row(csv_row)
                        team_idx_in_player_features = PLAYER_FEATURE_NAMES_UI.index('p_team')
                        plot_title = (f"TIdx:{selected_state_info['timeline_idx']}, OIdx:{original_data_idx} "
                                      f"(O:{selected_state_info['orange_true']}/{selected_state_info['orange_pred_label']}, "
                                      f"B:{selected_state_info['blue_true']}/{selected_state_info['blue_pred_label']})")
                        fig = plot_avg_positions_ui(player_feat_np, global_feat_np, plot_title, team_idx_in_player_features)
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error reconstructing/plotting features for original_idx {original_data_idx}: {e}")
                        st.error("Ensure CSV column names and norm constants in UI script match training script.")
                else:
                    st.error(f"Original data index {original_data_idx} is out of bounds for CSV (size: {len(original_csv_df)}).")
            else:
                 st.warning("Selected inspection index is out of range for the current filtered timeline.")


elif uploaded_json_file is None or uploaded_csv_file is None:
    st.info("Please upload both the JSON results file and the original CSV data file using the sidebar to begin analysis.")