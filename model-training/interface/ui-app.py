import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns # For heatmap
import os 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch_geometric.data import Data 
from torch_geometric.nn import GCNConv, global_mean_pool 

# --- Configuration (MUST MATCH YOUR CURRENT TRAINING SCRIPT) ---
NUM_PLAYERS = 6
PLAYER_FEATURES_COUNT = 13 
HIDDEN_DIM_CONFIG = 32    
NUM_TRACKED_BOOST_PADS_UI = 6 # From your training script
GLOBAL_FEATURE_DIM_COUNT = 3 + 3 + NUM_TRACKED_BOOST_PADS_UI + 1 + 1 

POS_MIN_X, POS_MAX_X = -4096, 4096; POS_MIN_Y, POS_MAX_Y = -6000, 6000; POS_MIN_Z, POS_MAX_Z = 0, 2044
VEL_MIN, VEL_MAX = -2300, 2300; BOOST_MIN, BOOST_MAX = 0, 100; BALL_VEL_MIN, BALL_VEL_MAX = -6000, 6000
BOOST_PAD_MIN, BOOST_PAD_MAX = 0, 10; DIST_MIN, DIST_MAX = 0, (8192**2 + 10240**2 + 2044**2)**(1/2)

PLAYER_FEATURE_NAMES_UI = [
    'p_pos_x', 'p_pos_y', 'p_pos_z', 'p_vel_x', 'p_vel_y', 'p_vel_z',
    'p_forward_x', 'p_forward_y', 'p_forward_z', 'p_boost_amount', 
    'p_team', 'p_alive', 'p_dist_to_ball'
] 
GLOBAL_FEATURE_NAMES_UI = [
    'ball_pos_x', 'ball_pos_y', 'ball_pos_z', 
    'ball_vel_x', 'ball_vel_y', 'ball_vel_z'
]
for i in range(NUM_TRACKED_BOOST_PADS_UI):
    GLOBAL_FEATURE_NAMES_UI.append(f'boost_pad_{i}_respawn')
GLOBAL_FEATURE_NAMES_UI.extend([
    'ball_hit_team_num', 'seconds_remaining'
])
assert len(PLAYER_FEATURE_NAMES_UI) == PLAYER_FEATURES_COUNT, "PLAYER_FEATURE_NAMES_UI length mismatch"
assert len(GLOBAL_FEATURE_NAMES_UI) == GLOBAL_FEATURE_DIM_COUNT, "GLOBAL_FEATURE_NAMES_UI length mismatch"
# --- End Configuration ---

# --- PyTorch Model Definition ---
class SafeRocketLeagueGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(PLAYER_FEATURES_COUNT, HIDDEN_DIM_CONFIG)
        self.bn1 = nn.BatchNorm1d(HIDDEN_DIM_CONFIG)
        self.conv2 = GCNConv(HIDDEN_DIM_CONFIG, HIDDEN_DIM_CONFIG)
        self.bn2 = nn.BatchNorm1d(HIDDEN_DIM_CONFIG)
        self.orange_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM_CONFIG + GLOBAL_FEATURE_DIM_COUNT, 32),
            nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )
        self.blue_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM_CONFIG + GLOBAL_FEATURE_DIM_COUNT, 32),
            nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, data): 
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = self.conv1(x, edge_index, edge_weight); x = self.bn1(x); x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight); x = self.bn2(x); x = F.relu(x)
        graph_embed = global_mean_pool(x, data.batch) 
        combined_features = torch.cat([graph_embed, data.global_features], dim=1)
        return self.orange_head(combined_features), self.blue_head(combined_features)
# --- End PyTorch Model Definition ---

# --- Session State Initialization ---
if 'highlight_test_set' not in st.session_state: st.session_state.highlight_test_set = True 
if 'selected_filtered_timeline_idx_canonical' not in st.session_state: st.session_state.selected_filtered_timeline_idx_canonical = 0
if 'csv_inspect_idx_canonical' not in st.session_state: st.session_state.csv_inspect_idx_canonical = 0
if 'current_node_grads_raw' not in st.session_state: st.session_state.current_node_grads_raw = None
if 'current_global_grads_raw' not in st.session_state: st.session_state.current_global_grads_raw = None
if 'current_edge_grads_raw' not in st.session_state: st.session_state.current_edge_grads_raw = None
if 'grads_for_state_key' not in st.session_state: st.session_state.grads_for_state_key = None
# --- End Session State Initialization ---

# --- Model Loading Function ---
@st.cache_resource
def load_pytorch_model(model_class, model_path, device_str="cpu", **model_args):
    if not os.path.exists(model_path): return None
    try:
        model_instance = model_class(**model_args) 
        checkpoint = torch.load(model_path, map_location=torch.device(device_str))
        state_dict_key = 'model_state' if 'model_state' in checkpoint else 'model_state_dict' if 'model_state_dict' in checkpoint else None
        state_dict = checkpoint[state_dict_key] if state_dict_key else checkpoint
        model_instance.load_state_dict(state_dict)
        model_instance.to(torch.device(device_str)); model_instance.eval()
        return model_instance
    except Exception as e: st.error(f"Error loading PyTorch model from {model_path}: {e}"); return None
# --- End Model Loading Function ---

# --- Gradient Calculation Function ---
@st.cache_data(show_spinner=False) 
def calculate_input_gradients(_model, player_features_np, global_features_np, 
                              edge_index_np, edge_weight_np, 
                              target_team='orange', device_str="cpu"):
    if _model is None: return None, None, None 
    device = torch.device(device_str)
    x_tensor = torch.tensor(player_features_np, dtype=torch.float32, device=device, requires_grad=True)
    _global_features_base = torch.tensor(global_features_np, dtype=torch.float32, device=device, requires_grad=True)
    global_features_tensor = _global_features_base.unsqueeze(0); global_features_tensor.retain_grad() 
    edge_index_tensor = torch.tensor(edge_index_np, dtype=torch.long, device=device)
    edge_weight_tensor = None
    if edge_weight_np is not None and edge_weight_np.size > 0:
        edge_weight_tensor = torch.tensor(edge_weight_np, dtype=torch.float32, device=device, requires_grad=True)
    batch_vector = torch.zeros(x_tensor.shape[0], dtype=torch.long, device=device)
    current_data = Data(x=x_tensor, edge_index=edge_index_tensor, edge_weight=edge_weight_tensor, 
                        global_features=global_features_tensor, batch=batch_vector)
    _model.zero_grad()
    try: orange_prob, blue_prob = _model(current_data)
    except Exception as e: st.error(f"Model forward pass error: {e}"); return None, None, None
    target_output_prob = orange_prob if target_team == 'orange' else blue_prob if target_team == 'blue' else None
    if target_output_prob is None or target_output_prob.nelement() == 0: return None, None, None
    if target_output_prob.dim() == 0 or target_output_prob.nelement() == 1: target_output_prob.backward()
    else: target_output_prob.backward(torch.ones_like(target_output_prob, device=device))

    node_grads_raw_np = x_tensor.grad.cpu().numpy() if x_tensor.grad is not None else None
    global_grads_raw_np = global_features_tensor.grad.cpu().numpy().squeeze(0) if global_features_tensor.grad is not None else None
    edge_weight_grads_raw_np = edge_weight_tensor.grad.cpu().numpy() if edge_weight_tensor is not None and edge_weight_tensor.grad is not None else None
    
    return node_grads_raw_np, global_grads_raw_np, edge_weight_grads_raw_np
# --- End Gradient Calculation Function ---

# --- Data Preprocessing Logic ---
def normalize_ui(val, min_val, max_val): return (val - min_val) / (max_val - min_val + 1e-8)
def reconstruct_features_from_csv_row(csv_row_series):
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
    
    # CORRECTED Global Feature Reconstruction
    current_global_features_list_ui = [
        normalize_ui(float(csv_row_series.get('ball_pos_x', 0)), POS_MIN_X, POS_MAX_X),
        normalize_ui(float(csv_row_series.get('ball_pos_y', 0)), POS_MIN_Y, POS_MAX_Y),
        normalize_ui(float(csv_row_series.get('ball_pos_z', 0)), POS_MIN_Z, POS_MAX_Z),
        normalize_ui(float(csv_row_series.get('ball_vel_x', 0)), BALL_VEL_MIN, BALL_VEL_MAX),
        normalize_ui(float(csv_row_series.get('ball_vel_y', 0)), BALL_VEL_MIN, BALL_VEL_MAX),
        normalize_ui(float(csv_row_series.get('ball_vel_z', 0)), BALL_VEL_MIN, BALL_VEL_MAX),
    ]
    for pad_idx in range(NUM_TRACKED_BOOST_PADS_UI):
        column_name = f'boost_pad_{pad_idx}_respawn'
        pad_respawn_time = float(csv_row_series.get(column_name, BOOST_PAD_MAX))
        current_global_features_list_ui.append(normalize_ui(pad_respawn_time, BOOST_PAD_MIN, BOOST_PAD_MAX))
    current_global_features_list_ui.extend([
        float(csv_row_series.get('ball_hit_team_num', 0)),
        normalized_seconds
    ])
    global_features_np = np.array(current_global_features_list_ui, dtype=np.float32)

    if player_features_np.shape != (NUM_PLAYERS, PLAYER_FEATURES_COUNT): raise ValueError(f"Reconstructed player features shape mismatch")
    if global_features_np.shape != (GLOBAL_FEATURE_DIM_COUNT,): raise ValueError(f"Reconstructed global features shape mismatch: Expected {(GLOBAL_FEATURE_DIM_COUNT,)}, Got {global_features_np.shape}")
    return player_features_np, global_features_np
# --- End Data Preprocessing Logic ---

# --- Plotting Function ---
def plot_avg_positions_ui(player_features_np, global_features_np, plot_title_suffix, team_id_feature_idx):
    avg_player_pos = player_features_np[:, :2]; avg_ball_pos = global_features_np[:2]
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.axhline(y=normalize_ui(POS_MAX_Y, POS_MIN_Y, POS_MAX_Y) - 0.02, color='orange', linestyle='--', xmin=0.35, xmax=0.65, lw=2, label="Orange Goal Area (High Y)")
    ax.axhline(y=normalize_ui(POS_MIN_Y, POS_MIN_Y, POS_MAX_Y) + 0.02, color='blue', linestyle='--', xmin=0.35, xmax=0.65, lw=2, label="Blue Goal Area (Low Y)")
    ax.set_xlim(normalize_ui(POS_MIN_X,POS_MIN_X,POS_MAX_X) - 0.05, normalize_ui(POS_MAX_X,POS_MIN_X,POS_MAX_X) + 0.05)
    ax.set_ylim(normalize_ui(POS_MIN_Y,POS_MIN_Y,POS_MAX_Y) - 0.05, normalize_ui(POS_MAX_Y,POS_MIN_Y,POS_MAX_Y) + 0.05)
    player_colors = {0.0: 'blue', 1.0: 'orange'}; player_label_prefix = "P"
    for i in range(NUM_PLAYERS):
        player_x, player_y = avg_player_pos[i, 0], avg_player_pos[i, 1]; player_team_id = 0.0
        if team_id_feature_idx < player_features_np.shape[1]: player_team_id = player_features_np[i, team_id_feature_idx]
        color = player_colors.get(player_team_id, 'gray')
        ax.scatter(player_x, player_y, s=150, color=color, edgecolors='black', alpha=0.9, label=f'{player_label_prefix}{i} (Team {int(player_team_id)})')
        text_color_on_dot = 'white' if color in ['blue', 'indigo', 'darkgreen', 'black', '#6f42c1'] else 'black'
        ax.text(player_x, player_y, f'{player_label_prefix}{i}', color=text_color_on_dot, ha='center', va='center', fontsize=9, fontweight='bold')
    ax.scatter(avg_ball_pos[0], avg_ball_pos[1], s=200, color='black', marker='o', label='Ball', edgecolors='white', linewidth=1.5, zorder=10)
    ax.set_title(f'State: {plot_title_suffix}'); ax.set_xlabel('X Position (Normalized)'); ax.set_ylabel('Y Position (Normalized)')
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize='small'); ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.7); fig.tight_layout(rect=[0, 0, 0.85, 1])
    return fig
# --- End Plotting Function ---

# --- Timeline String View Function ---
def get_timeline_strings_html(chrono_data, start_idx, end_idx, highlight_test_active):
    if not chrono_data or start_idx >= len(chrono_data) or start_idx < 0: return "<p>No data to display.</p>"
    subset_data = chrono_data[start_idx:end_idx]
    color_map_orange = {(0,0):"#28a745",(1,1):"#007bff",(0,1):"#dc3545",(1,0):"#ffc107"}; color_map_blue = {(0,0):"#17a2b8",(1,1):"#6f42c1",(0,1):"#fd7e14",(1,0):"#e83e8c"}
    default_text_color = "#333333"; html_lines = {"Orange True":"","Orange Pred":"","Blue True":"","Blue Pred":""}
    for i, item in enumerate(subset_data):
        actual_timeline_idx = item['timeline_idx']; ot, op = item['orange_true'], item['orange_pred_label']; bt, bp = item['blue_true'], item['blue_pred_label']
        split_type = item.get('split','unknown'); is_test = split_type == 'test'
        current_highlight_style = "background-color:#e9e9e9;" if is_test and highlight_test_active else ""
        tooltip_text = f"TIdx:{actual_timeline_idx}\nOIdx:{item['original_idx']}\nSet:{split_type.capitalize()}"
        def styled_data_cell(val,color,bold=False,title_text=""):
            style_parts=[f"color:{color};"]; 
            if bold:style_parts.append("font-weight:bold;"); 
            if current_highlight_style:style_parts.append(current_highlight_style)
            style_parts.append("display:inline-block;width:1em;text-align:center;padding:1px 0;"); safe_title_text=title_text.replace('"','"')
            return f"<span style='{' '.join(style_parts)}' title='{safe_title_text}'>{val}</span>"
        html_lines["Orange True"]+=styled_data_cell(ot,color_map_orange.get((ot,ot),default_text_color),bold=True,title_text=tooltip_text)
        html_lines["Orange Pred"]+=styled_data_cell(op,color_map_orange.get((ot,op),default_text_color),title_text=tooltip_text)
        html_lines["Blue True"]+=styled_data_cell(bt,color_map_blue.get((bt,bt),default_text_color),bold=True,title_text=tooltip_text)
        html_lines["Blue Pred"]+=styled_data_cell(bp,color_map_blue.get((bt,bp),default_text_color),title_text=tooltip_text)
    final_html_str = f"""<div style="font-family:'Courier New',Courier,monospace;font-size:14px;white-space:pre;overflow-x:auto;border:1px solid #ccc;padding:10px;background-color:#f8f9fa;color:{default_text_color};line-height:1.5em;">Orange True: {html_lines['Orange True']}\nOrange Pred: {html_lines['Orange Pred']}\nBlue True  : {html_lines['Blue True']}\nBlue Pred  : {html_lines['Blue Pred']}</div><small><i>Hover to see Idx/Set. Highlight:Test Set.</i></small>"""; return final_html_str
# --- End Timeline String View Function ---

# --- Helper to find goal indices ---
@st.cache_data 
def find_goal_indices(data, label_key_orange, label_key_blue, index_key_to_display):
    """
    Finds the last index in a sequence of 1s for goal labels.
    data: List of dictionaries, each representing a state.
    label_key_orange: Key for orange team's true goal label in the dict.
    label_key_blue: Key for blue team's true goal label in the dict.
    index_key_to_display: Key for the index value to store (e.g., 'timeline_idx' or 'original_idx').
    """
    orange_goal_event_indices = []
    blue_goal_event_indices = []
    
    n_states = len(data)
    if n_states == 0:
        return orange_goal_event_indices, blue_goal_event_indices

    # Process Orange Goals
    for i in range(n_states):
        current_orange_label = data[i].get(label_key_orange, 0)
        if current_orange_label == 1:
            # Check if it's the last '1' in a sequence for Orange
            is_last_orange_one = False
            if i == n_states - 1: # If it's the very last state in the dataset
                is_last_orange_one = True
            else:
                next_orange_label = data[i+1].get(label_key_orange, 0)
                if next_orange_label == 0: # Next state is not an orange goal anticipation
                    is_last_orange_one = True
            
            if is_last_orange_one:
                orange_goal_event_indices.append(data[i][index_key_to_display])

    # Process Blue Goals
    for i in range(n_states):
        current_blue_label = data[i].get(label_key_blue, 0)
        if current_blue_label == 1:
            # Check if it's the last '1' in a sequence for Blue
            is_last_blue_one = False
            if i == n_states - 1: # If it's the very last state
                is_last_blue_one = True
            else:
                next_blue_label = data[i+1].get(label_key_blue, 0)
                if next_blue_label == 0: # Next state is not a blue goal anticipation
                    is_last_blue_one = True
            
            if is_last_blue_one:
                blue_goal_event_indices.append(data[i][index_key_to_display])
                
    return orange_goal_event_indices, blue_goal_event_indices
# --- End Helper to find goal indices ---

# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("Rocket League Game State Visualizer")
app_mode = st.sidebar.radio("Select Mode:", ("Full Analysis (CSV + Model JSON)", "CSV-Only State Viewer"), key="app_mode_radio")

# --- Full Analysis Mode ---
if app_mode == "Full Analysis (CSV + Model JSON)":
    st.sidebar.header("Full Analysis Files & Settings")
    uploaded_json_file = st.sidebar.file_uploader("Upload Chronological JSON Results", type="json", key="json_full_uploader")
    uploaded_csv_file_full = st.sidebar.file_uploader("Upload Original Game Data CSV", type="csv", key="csv_full_uploader")
    MODEL_CHECKPOINT_PATH_UI = st.sidebar.text_input("Path to Model Checkpoint (.pth)", value="./checkpoints/model_checkpoint.pth", key="model_path_input_ui_full")
    DEVICE_UI = "cpu"
    if st.sidebar.button("Toggle Test Set Highlighting", key="toggle_highlight_button_full"): st.session_state.highlight_test_set = not st.session_state.highlight_test_set
    highlight_status_text = "ON" if st.session_state.get('highlight_test_set', True) else "OFF"; st.sidebar.caption(f"Test Set Highlighting: {highlight_status_text}")
    @st.cache_data 
    def load_json_data_full(file): return json.load(file)
    @st.cache_data 
    def load_csv_data_full(file): return pd.read_csv(file, low_memory=False)
    chrono_data_full = None; original_csv_df_full = None; trained_model_instance = None
    if uploaded_json_file: chrono_data_full = load_json_data_full(uploaded_json_file)
    if uploaded_csv_file_full: original_csv_df_full = load_csv_data_full(uploaded_csv_file_full)
    if MODEL_CHECKPOINT_PATH_UI and os.path.exists(MODEL_CHECKPOINT_PATH_UI):
        if 'SafeRocketLeagueGCN' in globals(): trained_model_instance = load_pytorch_model(SafeRocketLeagueGCN, MODEL_CHECKPOINT_PATH_UI, DEVICE_UI)
        else: st.sidebar.error("Model class 'SafeRocketLeagueGCN' not defined.")
    elif MODEL_CHECKPOINT_PATH_UI: st.sidebar.warning(f"Model checkpoint not found at: {MODEL_CHECKPOINT_PATH_UI}")

    if chrono_data_full and original_csv_df_full is not None and not original_csv_df_full.empty:
        st.success(f"Full Analysis: Loaded {len(chrono_data_full)} JSON states & {len(original_csv_df_full)} CSV rows.")
        if trained_model_instance: st.sidebar.success("Trained model loaded.")
        else: st.sidebar.warning("Model not loaded. Feature importance & Model Weights disabled.")
        with st.sidebar.expander("Goal Event Indices (JSON)", expanded=False):
            orange_goals_json, blue_goals_json = find_goal_indices(chrono_data_full, 'orange_true', 'blue_true', 'timeline_idx')
            st.write("**Orange Goal Timeline Indices:**"); st.dataframe(pd.DataFrame(orange_goals_json, columns=["Timeline Idx"]), height=150, use_container_width=True)
            st.write("**Blue Goal Timeline Indices:**"); st.dataframe(pd.DataFrame(blue_goals_json, columns=["Timeline Idx"]), height=150, use_container_width=True)
        
        filtered_chrono_data = chrono_data_full; available_splits = sorted(list(set(item.get('split', 'unknown') for item in chrono_data_full)))
        if len(available_splits) > 1: 
            selected_split_filter = st.sidebar.selectbox("Filter by Dataset Split:", options=["All"] + available_splits, index=0, key="split_filter_selectbox")
            if selected_split_filter != "All": filtered_chrono_data = [item for item in chrono_data_full if item.get('split') == selected_split_filter]
        st.sidebar.info(f"Displaying {len(filtered_chrono_data)} of {len(chrono_data_full)} states.")

        st.header("Timeline View")
        if not filtered_chrono_data: st.warning("No data matches current filter.")
        else:
            timeline_window_size=st.slider("Timeline Window Size",10,200,50,10,key="timeline_win_slider_full");max_start_idx=len(filtered_chrono_data)-timeline_window_size
            if max_start_idx<0:max_start_idx=0
            timeline_start_idx=st.slider("Timeline Start Index",0,max_start_idx,0,max(1,timeline_window_size//2),key="timeline_start_slider_full");timeline_end_idx=min(timeline_start_idx+timeline_window_size,len(filtered_chrono_data))
            st.markdown(get_timeline_strings_html(filtered_chrono_data,timeline_start_idx,timeline_end_idx,st.session_state.highlight_test_set),unsafe_allow_html=True)
            st.caption(f"Displaying states from filtered index {timeline_start_idx} to {timeline_end_idx-1}")

            st.header("Inspect Single State")
            max_inspect_idx = len(filtered_chrono_data) - 1
            if max_inspect_idx < 0: st.info("No states to inspect with current filters.")
            else:
                if 'selected_filtered_timeline_idx_canonical' not in st.session_state: st.session_state.selected_filtered_timeline_idx_canonical = 0
                if not (0 <= st.session_state.selected_filtered_timeline_idx_canonical <= max_inspect_idx): st.session_state.selected_filtered_timeline_idx_canonical = 0
                def full_analysis_stepper_changed(): st.session_state.selected_filtered_timeline_idx_canonical = st.session_state.full_analysis_stepper_key
                st.number_input(f"Enter Index from Current Timeline (0 to {max_inspect_idx})",0,max_inspect_idx,st.session_state.selected_filtered_timeline_idx_canonical,1,key="full_analysis_stepper_key",on_change=full_analysis_stepper_changed)
                inspect_idx_for_logic = st.session_state.selected_filtered_timeline_idx_canonical

                if 0 <= inspect_idx_for_logic < len(filtered_chrono_data):
                    state_info = filtered_chrono_data[inspect_idx_for_logic]; orig_csv_idx = state_info['original_idx'] 
                    split_display = state_info.get('split', 'unknown').capitalize()
                    st.subheader(f"State (TIdx:{state_info['timeline_idx']}, OIdx:{orig_csv_idx}, Set:{split_display})")
                    col1,col2=st.columns(2); 
                    with col1:st.write(f"**Orange:** T:`{state_info['orange_true']}`,P:`{state_info['orange_pred_label']}`(Prob:`{state_info['orange_pred_prob']:.3f}`)")
                    with col2:st.write(f"**Blue:** T:`{state_info['blue_true']}`,P:`{state_info['blue_pred_label']}`(Prob:`{state_info['blue_pred_prob']:.3f}`)")
                    
                    p_feat, g_feat = None, None
                    if 0 <= orig_csv_idx < len(original_csv_df_full):
                        csv_row = original_csv_df_full.iloc[orig_csv_idx]
                        try:
                            p_feat, g_feat = reconstruct_features_from_csv_row(csv_row)
                            team_idx = PLAYER_FEATURE_NAMES_UI.index('p_team')
                            plot_title = (f"TIdx:{state_info['timeline_idx']},OIdx:{orig_csv_idx} (O:{state_info['orange_true']}/{state_info['orange_pred_label']},B:{state_info['blue_true']}/{state_info['blue_pred_label']})")
                            fig = plot_avg_positions_ui(p_feat, g_feat, plot_title, team_idx); st.pyplot(fig)
                        except ValueError as ve: st.error(f"Data Error CSV idx {orig_csv_idx}: {ve}")
                        except Exception as e: st.error(f"Plotting Error CSV idx {orig_csv_idx}: {e}")
                    else: st.error(f"Original CSV idx {orig_csv_idx} out of bounds.")

                    # --- Feature Importance Section ---
                    if p_feat is not None and g_feat is not None and trained_model_instance:
                        st.markdown("---"); st.subheader("Feature Importance (Input Gradients)")
                        selectbox_key_grad_team = f"grad_team_select_{orig_csv_idx}"
                        target_team_for_grads = st.selectbox("Importance for which team's prediction?", ("orange", "blue"), key=selectbox_key_grad_team)
                        button_key_calc_grad = f"calc_grad_btn_{orig_csv_idx}"

                        if st.button("Calculate Feature Importance", key=button_key_calc_grad):
                            num_nodes_for_grad=p_feat.shape[0];edge_idx_list_grad,edge_w_list_grad=[],[]
                            p_feat_torch_temp=torch.from_numpy(p_feat).to(torch.device(DEVICE_UI))
                            for r_i in range(num_nodes_for_grad):
                                for r_j in range(num_nodes_for_grad):
                                    if r_i != r_j: 
                                        pos_player_i=p_feat_torch_temp[r_i,:3];pos_player_j=p_feat_torch_temp[r_j,:3]
                                        dist_grad=torch.norm(pos_player_i-pos_player_j);weight_grad=1.0/(1.0+dist_grad+1e-8) 
                                        edge_w_list_grad.append(weight_grad.item());edge_idx_list_grad.append([r_i,r_j])       
                            edge_idx_np_grad=np.array(edge_idx_list_grad,dtype=np.int64).T if edge_idx_list_grad else np.empty((2,0),dtype=np.int64)
                            edge_w_np_grad=np.array(edge_w_list_grad,dtype=np.float32) if edge_w_list_grad else np.array([],dtype=np.float32)
                            with st.spinner("Calculating..."):
                                n_grads_raw,g_grads_raw,e_grads_raw=calculate_input_gradients(trained_model_instance,p_feat,g_feat,edge_idx_np_grad,edge_w_np_grad,target_team=target_team_for_grads,device_str=DEVICE_UI)
                            current_grad_state_key=f"{orig_csv_idx}_{target_team_for_grads}"
                            st.session_state.current_node_grads_raw=n_grads_raw;st.session_state.current_global_grads_raw=g_grads_raw
                            st.session_state.current_edge_grads_raw=e_grads_raw;st.session_state.grads_for_state_key=current_grad_state_key
                            st.rerun() 
                        
                        selected_grad_team_for_display = st.session_state.get(selectbox_key_grad_team, "orange")
                        current_display_grad_state_key = f"{orig_csv_idx}_{selected_grad_team_for_display}"

                        if st.session_state.get('grads_for_state_key') == current_display_grad_state_key:
                            node_grads_raw_plot = st.session_state.current_node_grads_raw
                            global_grads_raw_plot = st.session_state.current_global_grads_raw
                            edge_grads_raw_plot = st.session_state.current_edge_grads_raw

                            if node_grads_raw_plot is not None:
                                st.write(f"**Per-Player Node Feature Gradients (for {selected_grad_team_for_display.capitalize()})**")
                                fig_nh, ax_nh = plt.subplots(figsize=(max(10,PLAYER_FEATURES_COUNT*0.7),max(4,NUM_PLAYERS*0.6)))
                                sns.heatmap(node_grads_raw_plot,yticklabels=[f"P{k}" for k in range(NUM_PLAYERS)],xticklabels=PLAYER_FEATURE_NAMES_UI,
                                            annot=True,fmt=".1e",cmap="coolwarm",center=0,ax=ax_nh,linewidths=.5)
                                ax_nh.set_title("Node Feature Gradients (Raw Values)");plt.xticks(rotation=45,ha="right");plt.yticks(rotation=0);fig_nh.tight_layout();st.pyplot(fig_nh)
                                st.caption("Positive gradient: increasing feature -> increases pred prob. Negative: decreases prob.")
                            
                            if global_grads_raw_plot is not None:
                                st.write(f"**Global Feature Gradients (for {selected_grad_team_for_display.capitalize()})**")
                                fig_gg,ax_gg=plt.subplots(figsize=(10,max(4,len(GLOBAL_FEATURE_NAMES_UI)*0.4)))
                                colors_gg=['tomato' if x<0 else 'mediumseagreen' for x in global_grads_raw_plot]
                                ax_gg.barh(GLOBAL_FEATURE_NAMES_UI,global_grads_raw_plot,color=colors_gg)
                                ax_gg.set_xlabel("Raw Gradient");ax_gg.set_title("Global Feature Gradients");ax_gg.invert_yaxis();fig_gg.tight_layout();st.pyplot(fig_gg)
                            
                            # Reconstruct edge_idx_np_grad and edge_w_np_grad again for display consistency with edge_grads_raw_plot
                            # This is needed if grads were loaded from session state without recalculating edges in this pass
                            temp_num_nodes_disp = p_feat.shape[0]
                            temp_edge_idx_list_disp, temp_edge_w_list_disp = [], []
                            temp_p_feat_torch_disp = torch.from_numpy(p_feat).to(torch.device(DEVICE_UI))
                            for r_i_disp in range(temp_num_nodes_disp):
                                for r_j_disp in range(temp_num_nodes_disp):
                                    if r_i_disp != r_j_disp:
                                        d_disp = torch.norm(temp_p_feat_torch_disp[r_i_disp,:3] - temp_p_feat_torch_disp[r_j_disp,:3])
                                        w_disp = 1.0 / (1.0 + d_disp + 1e-8)
                                        temp_edge_w_list_disp.append(w_disp.item())
                                        temp_edge_idx_list_disp.append([r_i_disp, r_j_disp])
                            current_edge_idx_np_disp = np.array(temp_edge_idx_list_disp,dtype=np.int64).T if temp_edge_idx_list_disp else np.empty((2,0),dtype=np.int64)
                            current_edge_weights_np_disp = np.array(temp_edge_w_list_disp,dtype=np.float32) if temp_edge_w_list_disp else np.array([],dtype=np.float32)

                            if edge_grads_raw_plot is not None and current_edge_idx_np_disp.shape[1] == len(edge_grads_raw_plot):
                                st.write(f"**Top Edge Weight Gradients (for {selected_grad_team_for_display.capitalize()})**")
                                edge_importances = []
                                for k_edge in range(len(edge_grads_raw_plot)):
                                    u,v=current_edge_idx_np_disp[0,k_edge],current_edge_idx_np_disp[1,k_edge]
                                    original_weight = current_edge_weights_np_disp[k_edge] if k_edge < len(current_edge_weights_np_disp) else np.nan
                                    edge_importances.append(((u,v),edge_grads_raw_plot[k_edge], original_weight))
                                sorted_edge_importances = sorted(edge_importances,key=lambda x:abs(x[1]),reverse=True)
                                df_edge_imp = pd.DataFrame(
                                    [(f"P{u}-P{v}",f"{imp:.4f}",f"{w:.3f}") for ((u,v),imp,w) in sorted_edge_importances[:10]],
                                    columns=["Edge","Raw Gradient","Original Weight"]
                                )
                                st.dataframe(df_edge_imp, use_container_width=True)
                            elif edge_grads_raw_plot is not None: st.caption("Edge grads available, but edge structure mismatch for display.")
                        elif st.session_state.get('grads_for_state_key') is not None:
                             st.info(f"Grads calculated for a different selection. Recalculate for '{selected_grad_team_for_display.capitalize()}'.")
                    elif not trained_model_instance: st.info("Provide model checkpoint for feature importance.")
    elif uploaded_json_file is None or uploaded_csv_file_full is None:
        st.info("For Full Analysis, please upload both JSON and original CSV.")

# --- CSV-Only State Viewer Mode ---
elif app_mode == "CSV-Only State Viewer":
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
                    fig = plot_avg_positions_ui(p_feat, g_feat, plot_title, team_idx); st.pyplot(fig)
                except ValueError as ve: st.error(f"Data Error for CSV idx {current_inspect_idx_for_logic}: {ve}")
                except KeyError as e: st.error(f"KeyError for CSV idx {current_inspect_idx_for_logic}: Missing column {e}.")
                except Exception as e: st.error(f"Error plotting for CSV idx {current_inspect_idx_for_logic}: {e}")
    elif uploaded_csv_file_viewer is None and app_mode == "CSV-Only State Viewer":
        st.info("For CSV-Only Viewer, please upload the original CSV data file.")

# --- Model Weights Inspection ---
st.sidebar.markdown("---") 
if trained_model_instance and app_mode == "Full Analysis (CSV + Model JSON)":
    with st.sidebar.expander("Inspect Model Weights", expanded=False):
        try:
            layer_names = [name for name, _ in trained_model_instance.named_parameters()]
            if not layer_names: st.write("No named parameters found.")
            else:
                selected_layer_name = st.selectbox("Select Layer to Inspect Weights:", layer_names, key="weight_layer_select")
                if selected_layer_name:
                    selected_weights = trained_model_instance.state_dict()[selected_layer_name].cpu().numpy()
                    st.write(f"**Weights for: `{selected_layer_name}`** (Shape: {selected_weights.shape})")
                    if selected_weights.ndim == 1: st.dataframe(pd.DataFrame(selected_weights.flatten(), columns=["Bias/Weight"]))
                    elif selected_weights.ndim == 2: 
                        if selected_weights.shape[0] > 30 or selected_weights.shape[1] > 30:
                             st.text("Matrix too large, showing stats & sample:")
                             st.write(f"Mean:{selected_weights.mean():.4f}, Std:{selected_weights.std():.4f}, Min:{selected_weights.min():.4f}, Max:{selected_weights.max():.4f}")
                             st.dataframe(pd.DataFrame(selected_weights[:min(10,selected_weights.shape[0]),:min(10,selected_weights.shape[1])]))
                        else:
                            fig_w,ax_w=plt.subplots(figsize=(max(6,selected_weights.shape[1]*0.5),max(4,selected_weights.shape[0]*0.5)))
                            sns.heatmap(selected_weights,annot=True,fmt=".1e",cmap="viridis",ax=ax_w,center=0)
                            ax_w.set_title(f"Weights: {selected_layer_name}");fig_w.tight_layout();st.pyplot(fig_w)
                    else: st.text("Weight tensor >2D, showing raw array."); st.write(selected_weights)
        except Exception as e: st.error(f"Error displaying model weights: {e}")
elif app_mode == "Full Analysis (CSV + Model JSON)": 
    st.sidebar.info("Load model in 'Full Analysis' mode to inspect weights.")
# --- End Model Weights Inspection ---