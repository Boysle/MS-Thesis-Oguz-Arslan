import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import os # Added for os.path.exists

import torch 
import torch.nn as nn # Import nn for model class
import torch.nn.functional as F 
from torch_geometric.data import Data 
from torch_geometric.nn import GCNConv, global_mean_pool # Import GCNConv, global_mean_pool

# --- Configuration (MUST MATCH YOUR TRAINING SCRIPT) ---
# These are used by data reconstruction and by the model class if it references them directly.
NUM_PLAYERS = 6
PLAYER_FEATURES_COUNT = 13 # Corresponds to PLAYER_FEATURES in training
HIDDEN_DIM_CONFIG = 32    # Corresponds to HIDDEN_DIM in training
GLOBAL_FEATURE_DIM_COUNT = 9 # Corresponds to GLOBAL_FEATURE_DIM in training

POS_MIN_X, POS_MAX_X = -4096, 4096
POS_MIN_Y, POS_MAX_Y = -6000, 6000
# ... (rest of your normalization bounds - make sure they are all here and correct) ...
POS_MIN_Z, POS_MAX_Z = 0, 2044
VEL_MIN, VEL_MAX = -2300, 2300
BOOST_MIN, BOOST_MAX = 0, 100
BALL_VEL_MIN, BALL_VEL_MAX = -6000, 6000
BOOST_PAD_MIN, BOOST_PAD_MAX = 0, 10
DIST_MIN, DIST_MAX = 0, (8192**2 + 10240**2 + 2044**2)**(1/2)


PLAYER_FEATURE_NAMES_UI = [
    'p_pos_x', 'p_pos_y', 'p_pos_z', 'p_vel_x', 'p_vel_y', 'p_vel_z',
    'p_forward_x', 'p_forward_y', 'p_forward_z', 'p_boost_amount', 
    'p_team', 'p_alive', 'p_dist_to_ball'
] # Length should be PLAYER_FEATURES_COUNT
GLOBAL_FEATURE_NAMES_UI = [
    'ball_pos_x', 'ball_pos_y', 'ball_pos_z', 'ball_vel_x', 'ball_vel_y', 'ball_vel_z',
    'boost_pad_0_respawn', 'ball_hit_team_num', 'seconds_remaining'
] # Length should be GLOBAL_FEATURE_DIM_COUNT
# --- End Configuration ---


# --- PyTorch Model Definition (Your Exact Class) ---
class SafeRocketLeagueGCN(nn.Module):
    def __init__(self):
        super().__init__()
        # Use the config constants defined above for clarity and consistency
        self.conv1 = GCNConv(PLAYER_FEATURES_COUNT, HIDDEN_DIM_CONFIG)
        self.bn1 = nn.BatchNorm1d(HIDDEN_DIM_CONFIG)
        self.conv2 = GCNConv(HIDDEN_DIM_CONFIG, HIDDEN_DIM_CONFIG)
        self.bn2 = nn.BatchNorm1d(HIDDEN_DIM_CONFIG)
        self.orange_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM_CONFIG + GLOBAL_FEATURE_DIM_COUNT, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.blue_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM_CONFIG + GLOBAL_FEATURE_DIM_COUNT, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, data): # Model expects a PyG Data object
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        graph_embed = global_mean_pool(x, data.batch) 
        combined_features = torch.cat([graph_embed, data.global_features], dim=1)
        return self.orange_head(combined_features), self.blue_head(combined_features)
# --- End PyTorch Model Definition ---


# --- Session State Initialization ---
# (Keep this as it was)
if 'highlight_test_set' not in st.session_state:
    st.session_state.highlight_test_set = True 
if 'selected_filtered_timeline_idx_canonical' not in st.session_state: # For Full Analysis mode inspector
     st.session_state.selected_filtered_timeline_idx_canonical = 0
if 'csv_inspect_idx_canonical' not in st.session_state: # For CSV-only mode inspector
    st.session_state.csv_inspect_idx_canonical = 0
# To store calculated gradients for the currently inspected item in Full Analysis mode
if 'current_node_grads' not in st.session_state:
    st.session_state.current_node_grads = None
if 'current_global_grads' not in st.session_state:
    st.session_state.current_global_grads = None
if 'grads_for_state_key' not in st.session_state: # Tracks which state grads are for
    st.session_state.grads_for_state_key = None
# --- End Session State Initialization ---


# --- Model Loading Function ---
@st.cache_resource
def load_pytorch_model(model_class, model_path, device_str="cpu", **model_args):
    if not os.path.exists(model_path):
        #st.error(f"Model checkpoint not found at: {model_path}") # Error shown by caller
        return None
    try:
        model_instance = model_class(**model_args) 
        checkpoint = torch.load(model_path, map_location=torch.device(device_str))
        state_dict_key = 'model_state' if 'model_state' in checkpoint else \
                         'model_state_dict' if 'model_state_dict' in checkpoint else None
        
        if state_dict_key:
            state_dict = checkpoint[state_dict_key]
        else: # Assume the checkpoint IS the state_dict
            state_dict = checkpoint
            
        model_instance.load_state_dict(state_dict)
        model_instance.to(torch.device(device_str))
        model_instance.eval()
        # st.success(f"Model loaded successfully from {model_path} to {device_str}.") # Success shown by caller
        return model_instance
    except Exception as e:
        st.error(f"Error loading PyTorch model from {model_path}: {e}")
        return None
# --- End Model Loading Function ---


# --- Gradient Calculation Function ---
@st.cache_data(show_spinner=False)
def calculate_input_gradients(_model, player_features_np, global_features_np,
                              edge_index_np, edge_weight_np,
                              target_team='orange', device_str="cpu"):
    if _model is None: return None, None, None 
    device = torch.device(device_str)
    
    # Player features (leaf tensor)
    x_tensor = torch.tensor(player_features_np, dtype=torch.float32, device=device, requires_grad=True)
    
    # Global features
    # Create the initial tensor that requires grad
    _global_features_base = torch.tensor(global_features_np, dtype=torch.float32, device=device, requires_grad=True)
    # Then unsqueeze. The result 'global_features_tensor' is non-leaf.
    global_features_tensor = _global_features_base.unsqueeze(0) 
    global_features_tensor.retain_grad() # <--- ADD THIS LINE

    # Edge index and weight
    edge_index_tensor = torch.tensor(edge_index_np, dtype=torch.long, device=device)
    edge_weight_tensor = None
    if edge_weight_np is not None and edge_weight_np.size > 0:
        edge_weight_tensor = torch.tensor(edge_weight_np, dtype=torch.float32, device=device, requires_grad=True) # Also make this require grad

    batch_vector = torch.zeros(x_tensor.shape[0], dtype=torch.long, device=device)
    current_data = Data(x=x_tensor, edge_index=edge_index_tensor, edge_weight=edge_weight_tensor,
                        global_features=global_features_tensor, batch=batch_vector) # Pass the unsqueezed tensor
    
    _model.zero_grad()
    try:
        orange_prob, blue_prob = _model(current_data)
    except Exception as e:
        st.error(f"Model forward pass error during gradient calculation: {e}")
        return None, None, None
        
    target_output_prob = orange_prob if target_team == 'orange' else blue_prob if target_team == 'blue' else None
    if target_output_prob is None or target_output_prob.nelement() == 0: return None, None, None
    
    if target_output_prob.dim() == 0 or target_output_prob.nelement() == 1:
        target_output_prob.backward()
    else:
        target_output_prob.backward(torch.ones_like(target_output_prob, device=device))

    node_grads_np = x_tensor.grad.abs().cpu().numpy() if x_tensor.grad is not None else None
    # Now global_features_tensor.grad should be populated
    global_grads_np = global_features_tensor.grad.abs().cpu().numpy().squeeze(0) if global_features_tensor.grad is not None else None
    edge_weight_grads_np = edge_weight_tensor.grad.abs().cpu().numpy() if edge_weight_tensor is not None and edge_weight_tensor.grad is not None else None
    
    return node_grads_np, global_grads_np, edge_weight_grads_np
# --- End Gradient Calculation Function ---


# --- Data Preprocessing Logic (Shared - Keep as is) ---
def normalize_ui(val, min_val, max_val): # ...
    return (val - min_val) / (max_val - min_val + 1e-8)

def reconstruct_features_from_csv_row(csv_row_series): # ...
    # (Keep this function exactly as it was, using the global config constants)
    player_features_list = []
    for i in range(NUM_PLAYERS):
        player_features_list.append([
            normalize_ui(float(csv_row_series.get(f'p{i}_pos_x', 0)), POS_MIN_X, POS_MAX_X),
            normalize_ui(float(csv_row_series.get(f'p{i}_pos_y', 0)), POS_MIN_Y, POS_MAX_Y),
            normalize_ui(float(csv_row_series.get(f'p{i}_pos_z', 0)), POS_MIN_Z, POS_MAX_Z),
            normalize_ui(float(csv_row_series.get(f'p{i}_vel_x', 0)), VEL_MIN, VEL_MAX),
            normalize_ui(float(csv_row_series.get(f'p{i}_vel_y', 0)), VEL_MIN, VEL_MAX),
            normalize_ui(float(csv_row_series.get(f'p{i}_vel_z', 0)), VEL_MIN, VEL_MAX),
            float(csv_row_series.get(f'p{i}_forward_x', 0)),
            float(csv_row_series.get(f'p{i}_forward_y', 0)),
            float(csv_row_series.get(f'p{i}_forward_z', 0)),
            normalize_ui(float(csv_row_series.get(f'p{i}_boost_amount', 0)), BOOST_MIN, BOOST_MAX),
            float(csv_row_series.get(f'p{i}_team', 0)),
            float(csv_row_series.get(f'p{i}_alive', 0)),
            normalize_ui(float(csv_row_series.get(f'p{i}_dist_to_ball', 0)), DIST_MIN, DIST_MAX)
        ])
    player_features_np = np.array(player_features_list, dtype=np.float32)

    seconds_remaining_val = float(csv_row_series.get('seconds_remaining', 0))
    normalized_seconds = normalize_ui(min(seconds_remaining_val, 300.0), 0, 300)

    global_features_np = np.array([
        normalize_ui(float(csv_row_series.get('ball_pos_x', 0)), POS_MIN_X, POS_MAX_X),
        normalize_ui(float(csv_row_series.get('ball_pos_y', 0)), POS_MIN_Y, POS_MAX_Y),
        normalize_ui(float(csv_row_series.get('ball_pos_z', 0)), POS_MIN_Z, POS_MAX_Z),
        normalize_ui(float(csv_row_series.get('ball_vel_x', 0)), BALL_VEL_MIN, BALL_VEL_MAX),
        normalize_ui(float(csv_row_series.get('ball_vel_y', 0)), BALL_VEL_MIN, BALL_VEL_MAX),
        normalize_ui(float(csv_row_series.get('ball_vel_z', 0)), BALL_VEL_MIN, BALL_VEL_MAX),
        normalize_ui(float(csv_row_series.get('boost_pad_0_respawn', 0)), BOOST_PAD_MIN, BOOST_PAD_MAX),
        float(csv_row_series.get('ball_hit_team_num', 0)),
        normalized_seconds
    ], dtype=np.float32)

    # Basic shape checks
    if player_features_np.shape != (NUM_PLAYERS, PLAYER_FEATURES_COUNT):
        # This error is now handled by the caller to avoid stopping the app if one row is bad
        raise ValueError(f"Reconstructed player features shape mismatch: Expected {(NUM_PLAYERS, PLAYER_FEATURES_COUNT)}, Got {player_features_np.shape}")
    if global_features_np.shape != (GLOBAL_FEATURE_DIM_COUNT,):
        raise ValueError(f"Reconstructed global features shape mismatch: Expected {(GLOBAL_FEATURE_DIM_COUNT,)}, Got {global_features_np.shape}")
    return player_features_np, global_features_np
# --- End Data Preprocessing Logic ---


# --- Plotting Function (Shared - Keep as is) ---
def plot_avg_positions_ui(player_features_np, global_features_np, plot_title_suffix, team_id_feature_idx): # ...
    # (Keep this function exactly as it was)
    avg_player_pos = player_features_np[:, :2]
    avg_ball_pos = global_features_np[:2]

    fig, ax = plt.subplots(figsize=(7, 9))
    ax.axhline(y=normalize_ui(POS_MAX_Y, POS_MIN_Y, POS_MAX_Y) - 0.02, color='orange', linestyle='--', xmin=0.35, xmax=0.65, lw=2, label="Orange Goal Area (High Y)")
    ax.axhline(y=normalize_ui(POS_MIN_Y, POS_MIN_Y, POS_MAX_Y) + 0.02, color='blue', linestyle='--', xmin=0.35, xmax=0.65, lw=2, label="Blue Goal Area (Low Y)")
    ax.set_xlim(normalize_ui(POS_MIN_X,POS_MIN_X,POS_MAX_X) - 0.05, normalize_ui(POS_MAX_X,POS_MIN_X,POS_MAX_X) + 0.05)
    ax.set_ylim(normalize_ui(POS_MIN_Y,POS_MIN_Y,POS_MAX_Y) - 0.05, normalize_ui(POS_MAX_Y,POS_MIN_Y,POS_MAX_Y) + 0.05)

    player_colors = {0.0: 'blue', 1.0: 'orange'}
    player_label_prefix = "P"

    for i in range(NUM_PLAYERS):
        player_x, player_y = avg_player_pos[i, 0], avg_player_pos[i, 1]
        player_team_id = 0.0
        if team_id_feature_idx < player_features_np.shape[1]:
            player_team_id = player_features_np[i, team_id_feature_idx]
        color = player_colors.get(player_team_id, 'gray')
        ax.scatter(player_x, player_y, s=150, color=color, edgecolors='black', alpha=0.9, label=f'{player_label_prefix}{i} (Team {int(player_team_id)})')
        text_color_on_dot = 'white' if color in ['blue', 'indigo', 'darkgreen', 'black', '#6f42c1'] else 'black'
        ax.text(player_x, player_y, f'{player_label_prefix}{i}', color=text_color_on_dot, ha='center', va='center', fontsize=9, fontweight='bold')

    ax.scatter(avg_ball_pos[0], avg_ball_pos[1], s=200, color='black', marker='o', label='Ball', edgecolors='white', linewidth=1.5, zorder=10)
    ax.set_title(f'State: {plot_title_suffix}')
    ax.set_xlabel('X Position (Normalized)')
    ax.set_ylabel('Y Position (Normalized)')
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize='small')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.7)
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    return fig
# --- End Plotting Function ---


# --- Timeline String View Function (For Full Analysis Mode - Keep as is) ---
def get_timeline_strings_html(chrono_data, start_idx, end_idx, highlight_test_active): # ...
    # (Keep this function exactly as it was)
    if not chrono_data or start_idx >= len(chrono_data) or start_idx < 0:
        return "<p>No data to display for this range.</p>"
    subset_data = chrono_data[start_idx:end_idx]
    color_map_orange = {(0, 0): "#28a745", (1, 1): "#007bff", (0, 1): "#dc3545", (1, 0): "#ffc107"}
    color_map_blue = {(0, 0): "#17a2b8", (1, 1): "#6f42c1", (0, 1): "#fd7e14", (1, 0): "#e83e8c"}
    default_text_color = "#333333"
    html_lines = {"Orange True": "", "Orange Pred": "", "Blue True": "", "Blue Pred": ""}

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
            style_parts.append("display:inline-block; width:1em; text-align:center; padding: 1px 0;")
            safe_title_text = title_text.replace('"', '"')
            return f"<span style='{' '.join(style_parts)}' title='{safe_title_text}'>{val}</span>"

        html_lines["Orange True"] += styled_data_cell(ot, color_map_orange.get((ot, ot), default_text_color), bold=True, title_text=tooltip_text)
        html_lines["Orange Pred"] += styled_data_cell(op, color_map_orange.get((ot, op), default_text_color), title_text=tooltip_text)
        html_lines["Blue True"] += styled_data_cell(bt, color_map_blue.get((bt, bt), default_text_color), bold=True, title_text=tooltip_text)
        html_lines["Blue Pred"] += styled_data_cell(bp, color_map_blue.get((bt, bp), default_text_color), title_text=tooltip_text)

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


# --- Helper to find goal indices (Keep as is) ---
@st.cache_data 
def find_goal_indices(data, label_key_orange, label_key_blue, index_key_to_display): # ...
    # (Keep this function exactly as it was)
    orange_goal_indices = [item[index_key_to_display] for item in data if item.get(label_key_orange) == 1]
    blue_goal_indices = [item[index_key_to_display] for item in data if item.get(label_key_blue) == 1]
    return orange_goal_indices, blue_goal_indices

# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("Rocket League Game State Visualizer")

app_mode = st.sidebar.radio("Select Mode:", ("Full Analysis (CSV + Model JSON)", "CSV-Only State Viewer"), key="app_mode_radio")

# --- Full Analysis Mode ---
if app_mode == "Full Analysis (CSV + Model JSON)":
    st.sidebar.header("Full Analysis Files & Settings")
    uploaded_json_file = st.sidebar.file_uploader("Upload Chronological JSON Results", type="json", key="json_full_uploader")
    uploaded_csv_file_full = st.sidebar.file_uploader("Upload Original Game Data CSV", type="csv", key="csv_full_uploader")
    
    MODEL_CHECKPOINT_PATH_UI = st.sidebar.text_input( # MODEL PATH INPUT
        "Path to Model Checkpoint (.pth)", 
        value="./checkpoints/model_checkpoint.pth", # Provide a sensible default
        key="model_path_input_ui_full"
    )
    DEVICE_UI = "cpu" # For UI, CPU is usually best

    if st.sidebar.button("Toggle Test Set Highlighting", key="toggle_highlight_button_full"):
        st.session_state.highlight_test_set = not st.session_state.highlight_test_set
    highlight_status_text = "ON" if st.session_state.get('highlight_test_set', True) else "OFF"
    st.sidebar.caption(f"Test Set Highlighting: {highlight_status_text}")

    @st.cache_data 
    def load_json_data_full(file): return json.load(file)
    @st.cache_data 
    def load_csv_data_full(file): return pd.read_csv(file, low_memory=False)

    chrono_data_full = None
    original_csv_df_full = None
    trained_model_instance = None # Initialize model instance variable

    if uploaded_json_file: chrono_data_full = load_json_data_full(uploaded_json_file)
    if uploaded_csv_file_full: original_csv_df_full = load_csv_data_full(uploaded_csv_file_full)
    
    # Attempt to load model if path is provided and seems valid
    if MODEL_CHECKPOINT_PATH_UI and os.path.exists(MODEL_CHECKPOINT_PATH_UI):
        if 'SafeRocketLeagueGCN' in globals(): # Check if class is defined
             trained_model_instance = load_pytorch_model(SafeRocketLeagueGCN, MODEL_CHECKPOINT_PATH_UI, DEVICE_UI)
        else:
            st.sidebar.error("Model class 'SafeRocketLeagueGCN' is not defined. Cannot load model.")
    elif MODEL_CHECKPOINT_PATH_UI: # Path provided but doesn't exist
        st.sidebar.warning(f"Model checkpoint not found at: {MODEL_CHECKPOINT_PATH_UI}")


    if chrono_data_full and original_csv_df_full is not None and not original_csv_df_full.empty:
        st.success(f"Full Analysis Mode: Loaded {len(chrono_data_full)} JSON states & {len(original_csv_df_full)} CSV rows.")
        if trained_model_instance:
            st.sidebar.success("Trained model loaded successfully.")
        else:
            st.sidebar.warning("Trained model not loaded. Feature importance calculation will be unavailable.")

        # ... (Goal Event Indices display logic - Keep as is) ...
        with st.sidebar.expander("Goal Event Indices (from JSON)", expanded=False):
            orange_goals_json, blue_goals_json = find_goal_indices(
                chrono_data_full, 'orange_true', 'blue_true', 'timeline_idx'
            )
            st.write("**Orange Goal Timeline Indices:**")
            st.dataframe(pd.DataFrame(orange_goals_json, columns=["Timeline Idx"]), height=150, use_container_width=True)
            st.write("**Blue Goal Timeline Indices:**")
            st.dataframe(pd.DataFrame(blue_goals_json, columns=["Timeline Idx"]), height=150, use_container_width=True)


        filtered_chrono_data = chrono_data_full 
        available_splits = sorted(list(set(item.get('split', 'unknown') for item in chrono_data_full)))
        if len(available_splits) > 1: 
            selected_split_filter = st.sidebar.selectbox("Filter by Dataset Split:", options=["All"] + available_splits, index=0, key="split_filter_selectbox")
            if selected_split_filter != "All":
                filtered_chrono_data = [item for item in chrono_data_full if item.get('split') == selected_split_filter]
        st.sidebar.info(f"Displaying {len(filtered_chrono_data)} of {len(chrono_data_full)} total states based on filter.")

        st.header("Timeline View")
        if not filtered_chrono_data:
            st.warning("No data matches the current filter criteria.")
        else:
            # ... (Timeline slider and markdown display logic - Keep as is) ...
            timeline_window_size = st.slider("Timeline Window Size", 10, 200, 50, 10, key="timeline_win_slider_full")
            max_start_idx = len(filtered_chrono_data) - timeline_window_size
            if max_start_idx < 0: max_start_idx = 0
            timeline_start_idx = st.slider("Timeline Start Index", 0, max_start_idx, 0, max(1, timeline_window_size//2), key="timeline_start_slider_full")
            timeline_end_idx = min(timeline_start_idx + timeline_window_size, len(filtered_chrono_data))
            st.markdown(get_timeline_strings_html(filtered_chrono_data, timeline_start_idx, timeline_end_idx, st.session_state.highlight_test_set), unsafe_allow_html=True)
            st.caption(f"Displaying states from filtered timeline index {timeline_start_idx} to {timeline_end_idx-1}")

            st.header("Inspect Single State")
            max_inspect_idx = len(filtered_chrono_data) - 1
            if max_inspect_idx < 0:
                st.info("No states available for inspection with current filters.")
            else:
                # --- Full Analysis State Inspection Logic with Callback ---
                if 'selected_filtered_timeline_idx_canonical' not in st.session_state:
                    st.session_state.selected_filtered_timeline_idx_canonical = 0
                if not (0 <= st.session_state.selected_filtered_timeline_idx_canonical <= max_inspect_idx):
                    st.session_state.selected_filtered_timeline_idx_canonical = 0
                def full_analysis_stepper_changed():
                    st.session_state.selected_filtered_timeline_idx_canonical = st.session_state.full_analysis_stepper_key
                
                st.number_input(f"Enter Index from Current Timeline (0 to {max_inspect_idx})", 0, max_inspect_idx, 
                                st.session_state.selected_filtered_timeline_idx_canonical, 1, 
                                key="full_analysis_stepper_key", on_change=full_analysis_stepper_changed)
                
                inspect_idx_for_logic = st.session_state.selected_filtered_timeline_idx_canonical

                if 0 <= inspect_idx_for_logic < len(filtered_chrono_data):
                    state_info = filtered_chrono_data[inspect_idx_for_logic] 
                    orig_csv_idx = state_info['original_idx'] 
                    split_display = state_info.get('split', 'unknown').capitalize()
                    st.subheader(f"State Details (Timeline Idx: {state_info['timeline_idx']}, Original CSV Idx: {orig_csv_idx}, Set: {split_display})")
                    col1, col2 = st.columns(2) # Display true/pred labels
                    with col1: st.write(f"**Orange Team:** True: `{state_info['orange_true']}`, Pred: `{state_info['orange_pred_label']}` (Prob: `{state_info['orange_pred_prob']:.3f}`)")
                    with col2: st.write(f"**Blue Team:** True: `{state_info['blue_true']}`, Pred: `{state_info['blue_pred_label']}` (Prob: `{state_info['blue_pred_prob']:.3f}`)")

                    p_feat, g_feat = None, None # Initialize
                    if 0 <= orig_csv_idx < len(original_csv_df_full):
                        csv_row = original_csv_df_full.iloc[orig_csv_idx]
                        try:
                            p_feat, g_feat = reconstruct_features_from_csv_row(csv_row) # Reconstruct for plotting & grads
                            team_idx = PLAYER_FEATURE_NAMES_UI.index('p_team')
                            plot_title = (f"TIdx:{state_info['timeline_idx']}, OIdx:{orig_csv_idx} "
                                          f"(O:{state_info['orange_true']}/{state_info['orange_pred_label']}, "
                                          f"B:{state_info['blue_true']}/{state_info['blue_pred_label']})")
                            fig = plot_avg_positions_ui(p_feat, g_feat, plot_title, team_idx)
                            st.pyplot(fig)
                        except ValueError as ve: # Catch specific error from reconstruct
                            st.error(f"Data Error for CSV idx {orig_csv_idx}: {ve}")
                        except Exception as e: st.error(f"Error plotting for CSV idx {orig_csv_idx}: {e}")
                    else: st.error(f"Original CSV index {orig_csv_idx} out of bounds.")

                    # --- Feature Importance Section ---
                    if p_feat is not None and g_feat is not None and trained_model_instance:
                        st.markdown("---") 
                        st.subheader("Feature Importance (Input Gradients)")
                        
                        # Selectbox to choose the team
                        # The key for this selectbox itself needs to be stable
                        selectbox_key_grad_team = f"grad_team_select_{state_info['original_idx']}"
                        target_team_for_grads = st.selectbox(
                            "Importance for which team's prediction?", 
                            ("orange", "blue"), 
                            key=selectbox_key_grad_team 
                        )
                        
                        # Button key is now simpler and doesn't depend on the selectbox's current value
                        button_key_calc_grad = f"calc_grad_btn_{state_info['original_idx']}"
                        if st.button("Calculate Feature Importance", key=button_key_calc_grad):
                            # target_team_for_grads is read here, AFTER the selectbox has been processed
                            # and its value is stable for this run.
                            
                            # ... (edge reconstruction logic as before) ...
                            num_nodes_for_grad = p_feat.shape[0]
                            # ... (rest of edge_idx_np_grad, edge_w_np_grad creation) ...
                            edge_idx_list_grad = []
                            edge_w_list_grad = []
                            p_feat_torch_temp = torch.from_numpy(p_feat).to(torch.device(DEVICE_UI)) 

                            for r_i in range(num_nodes_for_grad):
                                for r_j in range(num_nodes_for_grad):
                                    if r_i != r_j: 
                                        pos_player_i = p_feat_torch_temp[r_i, :3]
                                        pos_player_j = p_feat_torch_temp[r_j, :3]
                                        dist_grad = torch.norm(pos_player_i - pos_player_j)
                                        weight_grad = 1.0 / (1.0 + dist_grad + 1e-8) 
                                        edge_w_list_grad.append(weight_grad.item()) 
                                        edge_idx_list_grad.append([r_i, r_j])       
                            
                            if edge_idx_list_grad:
                                edge_idx_np_grad = np.array(edge_idx_list_grad, dtype=np.int64).T 
                                edge_w_np_grad = np.array(edge_w_list_grad, dtype=np.float32)
                            else: 
                                edge_idx_np_grad = np.empty((2,0), dtype=np.int64)
                                edge_w_np_grad = np.array([], dtype=np.float32)


                            with st.spinner("Calculating gradients..."):
                                node_grads, global_grads, edge_grads = calculate_input_gradients(
                                    trained_model_instance, p_feat, g_feat,
                                    edge_idx_np_grad, edge_w_np_grad, 
                                    target_team=target_team_for_grads, # Use the value from selectbox
                                    device_str=DEVICE_UI
                                )
                            # current_grad_state_key now uses the selected target_team_for_grads
                            current_grad_state_key = f"{state_info['original_idx']}_{target_team_for_grads}"
                            st.session_state.current_node_grads = node_grads
                            st.session_state.current_global_grads = global_grads
                            st.session_state.current_edge_grads = edge_grads 
                            st.session_state.grads_for_state_key = current_grad_state_key
                            st.rerun() 
                        
                        # Plotting logic (retrieves from session state)
                        # The key for retrieving from session state should use the selectbox's current value
                        selected_grad_team_for_display = st.session_state.get(selectbox_key_grad_team, "orange") # Use selectbox's key
                        current_display_grad_state_key = f"{state_info['original_idx']}_{selected_grad_team_for_display}"
                        
                        # Display plots if grads for current state/team are in session_state
                        if st.session_state.get('grads_for_state_key') == current_display_grad_state_key:
                            node_grads_to_plot = st.session_state.current_node_grads
                            global_grads_to_plot = st.session_state.current_global_grads

                            if node_grads_to_plot is not None:
                                st.write(f"**Node Feature Importances (Avg Abs Gradient for {target_team_for_grads.capitalize()})**")
                                avg_node_grads = np.mean(node_grads_to_plot, axis=0) 
                                fig_ng, ax_ng = plt.subplots(figsize=(10, max(4, len(PLAYER_FEATURE_NAMES_UI) * 0.4)))
                                ax_ng.barh(PLAYER_FEATURE_NAMES_UI, avg_node_grads, color='skyblue')
                                ax_ng.set_xlabel("Mean Absolute Gradient"); ax_ng.set_title("Average Node Feature Importance"); ax_ng.invert_yaxis(); fig_ng.tight_layout()
                                st.pyplot(fig_ng)
                            if global_grads_to_plot is not None:
                                st.write(f"**Global Feature Importances (Abs Gradient for {target_team_for_grads.capitalize()})**")
                                fig_gg, ax_gg = plt.subplots(figsize=(10, max(4, len(GLOBAL_FEATURE_NAMES_UI) * 0.4)))
                                ax_gg.barh(GLOBAL_FEATURE_NAMES_UI, global_grads_to_plot, color='lightcoral')
                                ax_gg.set_xlabel("Absolute Gradient"); ax_gg.set_title("Global Feature Importance"); ax_gg.invert_yaxis(); fig_gg.tight_layout()
                                st.pyplot(fig_gg)
                    elif not trained_model_instance:
                        st.info("Provide a valid model checkpoint path in the sidebar to calculate feature importance.")
                    # --- End Feature Importance Section ---
    elif uploaded_json_file is None or uploaded_csv_file_full is None:
        st.info("For Full Analysis, please upload both JSON results and the original CSV data file.")

# --- CSV-Only State Viewer Mode ---
elif app_mode == "CSV-Only State Viewer":
    # ... (Keep this mode's logic as it was, it doesn't involve model loading or gradients) ...
    st.sidebar.header("CSV-Only Viewer File")
    uploaded_csv_file_viewer = st.sidebar.file_uploader("Upload Original Game Data CSV", type="csv", key="csv_viewer_uploader")
    @st.cache_data
    def load_csv_data_viewer(file): return pd.read_csv(file, low_memory=False)
    original_csv_df_viewer = None
    if uploaded_csv_file_viewer: original_csv_df_viewer = load_csv_data_viewer(uploaded_csv_file_viewer)

    if original_csv_df_viewer is not None and not original_csv_df_viewer.empty:
        st.success(f"CSV-Only Mode: Loaded CSV with {len(original_csv_df_viewer)} rows.")
        with st.sidebar.expander("Goal Event Indices (from CSV)", expanded=False):
            @st.cache_data
            def get_csv_list_of_dicts(_df):
                df_with_idx = _df.copy(); df_with_idx['original_idx'] = df_with_idx.index
                return df_with_idx.to_dict('records')
            csv_data_list_of_dicts = get_csv_list_of_dicts(original_csv_df_viewer)
            orange_goals_csv, blue_goals_csv = find_goal_indices(csv_data_list_of_dicts, 'team_1_goal_prev_5s', 'team_0_goal_prev_5s', 'original_idx')
            st.write("**Orange Goal CSV Row Indices:**"); st.dataframe(pd.DataFrame(orange_goals_csv, columns=["CSV Row Idx"]), height=150, use_container_width=True)
            st.write("**Blue Goal CSV Row Indices:**"); st.dataframe(pd.DataFrame(blue_goals_csv, columns=["CSV Row Idx"]), height=150, use_container_width=True)

        st.header("Inspect CSV State by Row Index")
        max_csv_idx = len(original_csv_df_viewer) - 1
        if max_csv_idx < 0: st.warning("Uploaded CSV is empty.")
        else:
            if 'csv_inspect_idx_canonical' not in st.session_state: st.session_state.csv_inspect_idx_canonical = 0
            if not (0 <= st.session_state.csv_inspect_idx_canonical <= max_csv_idx): st.session_state.csv_inspect_idx_canonical = 0
            def csv_stepper_changed_viewer(): st.session_state.csv_inspect_idx_canonical = st.session_state.csv_stepper_key_viewer
            st.number_input(f"Enter CSV Row Index (0 to {max_csv_idx})", 0, max_csv_idx, st.session_state.csv_inspect_idx_canonical, 1, key="csv_stepper_key_viewer", on_change=csv_stepper_changed_viewer)
            current_inspect_idx_for_logic = st.session_state.csv_inspect_idx_canonical

            if 0 <= current_inspect_idx_for_logic < len(original_csv_df_viewer):
                csv_row = original_csv_df_viewer.iloc[current_inspect_idx_for_logic]
                st.subheader(f"Displaying State from CSV Row: {current_inspect_idx_for_logic}")
                blue_goal_raw = csv_row.get('team_0_goal_prev_5s', 'N/A'); orange_goal_raw = csv_row.get('team_1_goal_prev_5s', 'N/A')
                st.write(f"Raw CSV Goal Labels (contextual): Blue Goal (team_0_...): `{blue_goal_raw}`, Orange Goal (team_1_...): `{orange_goal_raw}`")
                try:
                    p_feat, g_feat = reconstruct_features_from_csv_row(csv_row)
                    team_idx = PLAYER_FEATURE_NAMES_UI.index('p_team')
                    plot_title = f"CSV Row: {current_inspect_idx_for_logic}"
                    fig = plot_avg_positions_ui(p_feat, g_feat, plot_title, team_idx)
                    st.pyplot(fig)
                except ValueError as ve: st.error(f"Data Error for CSV idx {current_inspect_idx_for_logic}: {ve}")
                except KeyError as e: st.error(f"KeyError for CSV idx {current_inspect_idx_for_logic}: Missing column {e}.")
                except Exception as e: st.error(f"Error plotting for CSV idx {current_inspect_idx_for_logic}: {e}")
    elif uploaded_csv_file_viewer is None and app_mode == "CSV-Only State Viewer":
        st.info("For CSV-Only Viewer, please upload the original CSV data file.")