import pandas as pd
import logging
import numpy as np
from pathlib import Path
import time
import sys
from pathlib import Path

# ==================================================================
# Configuration
# ==================================================================
# --- INPUT: Path to the FOLDER containing the chunked CSV files ---
DATASET_FOLDER_PATH = Path(r"E:\\Raw RL Esports Replays\\Big Replay Dataset\\split_dataset")

# --- Validation Parameters ---
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
GOAL_ANTICIPATION_WINDOW_SECONDS = 5.0
DATASET_HERTZ = 5 # IMPORTANT: Set this to match the HZ of the dataset file

# --- Validation Parameters for Field Shape ---
BOUNDARY_TOLERANCE = 15 # A small tolerance for all checks

# Main field dimensions
SIDE_WALL_X = 4096.0
BACK_WALL_Y = 5120.0
CEILING_Z = 2044.0
FLOOR_Z = 0.0

# Corner cut-off plane
CORNER_INTERSECT_XY = 8064.0

# Goal dimensions
GOAL_WIDTH_X = 892.755
GOAL_HEIGHT_Z = 642.775
GOAL_DEPTH_Y = 880.0 # From back wall (5120) to back of goal (5120+880=6000)

# Special case for how deep the ball can go (it's smaller than the goal depth)
# Using your previous value of 5120 + 182.5 = 5302.5
MAX_BALL_Y_IN_GOAL = 5302.5

# --- Expected Schema ---
# This schema defines the expected columns and their data types for validation.
EXPECTED_SCHEMA = {
    'replay_id': 'category', 'blue_score': 'int64', 'orange_score': 'int64', 'score_difference': 'int64',
    'seconds_remaining': 'int64',
    'p0_pos_x': 'float64', 'p0_pos_y': 'float64', 'p0_pos_z': 'float64',
    'p0_vel_x': 'float64', 'p0_vel_y': 'float64', 'p0_vel_z': 'float64',
    'p0_boost_amount': 'float64', 'p0_team': 'int64',
    'p1_pos_x': 'float64', 'p1_pos_y': 'float64', 'p1_pos_z': 'float64',
    'p1_vel_x': 'float64', 'p1_vel_y': 'float64', 'p1_vel_z': 'float64',
    'p1_boost_amount': 'float64', 'p1_team': 'int64',
    'p2_pos_x': 'float64', 'p2_pos_y': 'float64', 'p2_pos_z': 'float64',
    'p2_vel_x': 'float64', 'p2_vel_y': 'float64', 'p2_vel_z': 'float64',
    'p2_boost_amount': 'float64', 'p2_team': 'int64',
    'p3_pos_x': 'float64', 'p3_pos_y': 'float64', 'p3_pos_z': 'float64',
    'p3_vel_x': 'float64', 'p3_vel_y': 'float64', 'p3_vel_z': 'float64',
    'p3_boost_amount': 'float64', 'p3_team': 'int64',
    'p4_pos_x': 'float64', 'p4_pos_y': 'float64', 'p4_pos_z': 'float64',
    'p4_vel_x': 'float64', 'p4_vel_y': 'float64', 'p4_vel_z': 'float64',
    'p4_boost_amount': 'float64', 'p4_team': 'int64',
    'p5_pos_x': 'float64', 'p5_pos_y': 'float64', 'p5_pos_z': 'float64',
    'p5_vel_x': 'float64', 'p5_vel_y': 'float64', 'p5_vel_z': 'float64',
    'p5_boost_amount': 'float64', 'p5_team': 'int64',
    'ball_pos_x': 'float64', 'ball_pos_y': 'float64', 'ball_pos_z': 'float64',
    'ball_vel_x': 'float64', 'ball_vel_y': 'float64', 'ball_vel_z': 'float64',
    'ball_hit_team_num': 'float64',
    'p0_alive': 'int64', 'p1_alive': 'int64', 'p2_alive': 'int64',
    'p3_alive': 'int64', 'p4_alive': 'int64', 'p5_alive': 'int64',
    'p0_forward_x': 'float64', 'p0_forward_y': 'float64', 'p0_forward_z': 'float64',
    'p1_forward_x': 'float64', 'p1_forward_y': 'float64', 'p1_forward_z': 'float64',
    'p2_forward_x': 'float64', 'p2_forward_y': 'float64', 'p2_forward_z': 'float64',
    'p3_forward_x': 'float64', 'p3_forward_y': 'float64', 'p3_forward_z': 'float64',
    'p4_forward_x': 'float64', 'p4_forward_y': 'float64', 'p4_forward_z': 'float64',
    'p5_forward_x': 'float64', 'p5_forward_y': 'float64', 'p5_forward_z': 'float64',
    'boost_pad_0_respawn': 'float64', 'boost_pad_1_respawn': 'float64',
    'boost_pad_2_respawn': 'float64', 'boost_pad_3_respawn': 'float64',
    'boost_pad_4_respawn': 'float64', 'boost_pad_5_respawn': 'float64',
    'team_0_goal_in_event_window': 'int64', 'team_1_goal_in_event_window': 'int64',
    'p0_dist_to_ball': 'float64', 'p1_dist_to_ball': 'float64', 'p2_dist_to_ball': 'float64',
    'p3_dist_to_ball': 'float64', 'p4_dist_to_ball': 'float64', 'p5_dist_to_ball': 'float64',
    'time': 'float64' # 'time' column is required for the labeling logic validation
}

# ==================================================================
# Logging and Summary Functions
# ==================================================================
def configure_logging():
    """Set up logging to save the log file next to the script."""
    
    # Get the directory where the script itself is located
    script_dir = Path(__file__).resolve().parent
    
    # Define the full, absolute path for the log file
    log_file_path = script_dir / "dataset_validation_results_folder.log"
    
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        handlers=[
            # Use the absolute path here
            logging.FileHandler(log_file_path, mode='w'),
            logging.StreamHandler()
        ]
    )
    # Add a log message so you know exactly where it's saving
    print(f"--- Log file will be saved to: {log_file_path} ---")

def print_final_summary(results: dict, total_time: float):
    """Prints a clean, formatted summary of all validation checks."""
    summary_lines = [
        "\n" + "="*50,
        "          DATASET VALIDATION SUMMARY",
        "="*50
    ]
    
    overall_status = "PASSED"
    for check_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        summary_lines.append(f"{check_name:<35} | {status}")
        if not result:
            overall_status = "FAILED"
            
    summary_lines.extend([
        "-" * 50,
        f"Overall Validation Status: {overall_status}",
        f"Total Validation Time: {total_time:.2f} seconds",
        "=" * 50
    ])
    
    for line in summary_lines:
        logging.info(line)

# ==================================================================
# Validation Functions
# ==================================================================

def validate_schema(df: pd.DataFrame, expected_schema: dict) -> bool:
    """Validates column presence, naming, and data types."""
    if df is None: return False
    logging.info("--- Validating Dataset Schema ---")
    overall_success = True
    actual_cols_set = set(df.columns)
    expected_cols_set = set(expected_schema.keys())
    
    missing_cols = expected_cols_set - actual_cols_set
    if missing_cols:
        logging.error(f"FAILED: MISSING expected columns: {sorted(list(missing_cols))}")
        overall_success = False
    
    extra_cols = actual_cols_set - expected_cols_set
    if extra_cols:
        logging.warning(f"Found UNEXPECTED columns: {sorted(list(extra_cols))}")

    for col_name, expected_dtype in expected_schema.items():
        if col_name in df.columns:
            actual_dtype = str(df[col_name].dtype)
            if actual_dtype != expected_dtype:
                logging.error(f"FAILED: Column '{col_name}' DType Mismatch! Expected: {expected_dtype}, Actual: {actual_dtype}")
                overall_success = False
    
    if overall_success:
        logging.info("PASSED: Schema validation.")
    return overall_success    

def validate_missing_values(df: pd.DataFrame) -> bool:
    """Checks for any NaN values in the DataFrame."""
    if df is None: return False
    logging.info("--- Validating Missing Values (NaNs) ---")
    nan_summary = df.isnull().sum()
    columns_with_nans = nan_summary[nan_summary > 0]
    if columns_with_nans.empty:
        logging.info("PASSED: No NaN values found.")
        return True
    else:
        logging.error("FAILED: NaN values found!")
        for col, count in columns_with_nans.items():
            logging.error(f"  - Column '{col}': {count} NaNs")
        return False

def validate_infinity_values(df: pd.DataFrame) -> bool:
    """Checks for any infinity values in the DataFrame."""
    if df is None: return False
    logging.info("--- Validating Infinity Values ---")
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty:
        logging.info("No numeric columns to check for infinity values.")
        return True
    is_inf_df = numeric_df.apply(np.isinf)
    inf_summary = is_inf_df.sum()
    columns_with_inf = inf_summary[inf_summary > 0]
    if columns_with_inf.empty:
        logging.info("PASSED: No infinity values found.")
        return True
    else:
        logging.error("FAILED: Infinity values found!")
        for col, count in columns_with_inf.items():
            logging.error(f"  - Column '{col}': {count} infinity values")
        return False

def validate_playable_area(df: pd.DataFrame, max_examples_to_log: int = 5) -> bool:
    """
    Performs an advanced, multi-zone validation of player and ball positions.

    A position is valid if it is in ANY of these zones:
    1. The main field area (excluding corner cuts).
    2. The blue goal area.
    3. The orange goal area.
    """
    if df is None: return False
    logging.info("--- Validating Playable Area (Advanced Multi-Zone Method) ---")
    overall_success = True
    
    subjects = [f'p{i}' for i in range(6)] + ['ball']
    
    for subject in subjects:
        x_col, y_col, z_col = f"{subject}_pos_x", f"{subject}_pos_y", f"{subject}_pos_z"
        
        if not all(c in df.columns for c in [x_col, y_col, z_col]):
            logging.warning(f"Skipping playable area validation for '{subject}' due to missing columns.")
            continue

        # --- Define each valid zone using boolean masks ---

        # Zone 1: Main Field (Floor to Ceiling, excluding corner cuts)
        on_main_field_xy = (df[x_col].abs() <= SIDE_WALL_X + BOUNDARY_TOLERANCE) & \
                           (df[y_col].abs() <= BACK_WALL_Y + BOUNDARY_TOLERANCE)
        is_not_in_corner_cut = (df[x_col].abs() + df[y_col].abs() <= CORNER_INTERSECT_XY + BOUNDARY_TOLERANCE)
        on_main_field_z = (df[z_col] >= FLOOR_Z - BOUNDARY_TOLERANCE) & \
                          (df[z_col] <= CEILING_Z + BOUNDARY_TOLERANCE)
        
        is_in_main_field_zone = on_main_field_xy & is_not_in_corner_cut & on_main_field_z

        # --- Goal Zone Logic ---
        in_goal_width = df[x_col].abs() <= GOAL_WIDTH_X + BOUNDARY_TOLERANCE
        in_goal_height = (df[z_col] >= FLOOR_Z - BOUNDARY_TOLERANCE) & \
                         (df[z_col] <= GOAL_HEIGHT_Z + BOUNDARY_TOLERANCE)

        # For the ball, use the special Y-depth limit. For players, use the full goal depth.
        max_y_depth = MAX_BALL_Y_IN_GOAL if subject == 'ball' else (BACK_WALL_Y + GOAL_DEPTH_Y)

        # Zone 2: Blue Goal Area
        in_blue_goal_depth = (df[y_col] >= -max_y_depth - BOUNDARY_TOLERANCE) & \
                             (df[y_col] <= -BACK_WALL_Y + BOUNDARY_TOLERANCE)
        is_in_blue_goal_zone = in_goal_width & in_goal_height & in_blue_goal_depth
        
        # Zone 3: Orange Goal Area
        in_orange_goal_depth = (df[y_col] >= BACK_WALL_Y - BOUNDARY_TOLERANCE) & \
                               (df[y_col] <= max_y_depth + BOUNDARY_TOLERANCE)
        is_in_orange_goal_zone = in_goal_width & in_goal_height & in_orange_goal_depth

        # A position is valid if it is in ANY of the valid zones.
        is_valid_position = is_in_main_field_zone | is_in_blue_goal_zone | is_in_orange_goal_zone

        # Outliers are any rows where the position is NOT valid.
        outliers = df[~is_valid_position]

        if not outliers.empty:
            overall_success = False
            num_outliers = len(outliers)
            logging.error(f"FAILED: Subject '{subject}' has {num_outliers} positions outside of any valid zone (main field or goals).")
            logging.error(f"  -> First {min(num_outliers, max_examples_to_log)} examples:")
            for index, row in outliers.head(max_examples_to_log).iterrows():
                logging.error(f"    - Replay '{row['replay_id']}', Index {index}: (X={row[x_col]:.0f}, Y={row[y_col]:.0f}, Z={row[z_col]:.0f})")

    if overall_success:
        logging.info("PASSED: All player and ball positions are within the defined playable zones.")
    else:
        logging.error("FAILED: One or more subjects had positions outside the defined playable zones.")
        
    return overall_success



def validate_boost_amount(df: pd.DataFrame) -> bool:
    """Checks if player boost amounts are within the range [0, 100]."""
    if df is None: return False
    logging.info("--- Validating Player Boost Amounts ---")
    overall_success = True
    for i in range(6):
        col_name = f"p{i}_boost_amount"
        if col_name in df.columns:
            outliers = df[(df[col_name] < 0.0) | (df[col_name] > 100.0)]
            if not outliers.empty:
                overall_success = False
                logging.error(f"FAILED: Column '{col_name}' has {len(outliers)} values outside range [0, 100].")
    if overall_success:
        logging.info("PASSED: All boost amounts are within the expected range.")
    return overall_success

def validate_team_composition(df: pd.DataFrame) -> bool:
    """Validates that team identifiers are 0 or 1 and composition is 3v3."""
    if df is None: return False
    logging.info("--- Validating Team Composition and Identifiers ---")
    team_cols = [f"p{i}_team" for i in range(6)]
    if not all(col in df.columns for col in team_cols):
        logging.error("FAILED: Missing one or more player team columns.")
        return False
    all_team_values = pd.concat([df[col] for col in team_cols]).unique()
    if any(v not in [0, 1] for v in all_team_values):
        logging.error("FAILED: Invalid team ID(s) found. Expected only 0 or 1.")
        return False
    invalid_composition_rows = df[df[team_cols].sum(axis=1) != 3]
    if not invalid_composition_rows.empty:
        logging.error(f"FAILED: {len(invalid_composition_rows)} rows do not have a 3v3 composition.")
        return False
    logging.info("PASSED: Team composition is valid.")
    return True

def validate_player_alive_status(df: pd.DataFrame) -> bool:
    """Validates that player alive status is 0 or 1."""
    if df is None: return False
    logging.info("--- Validating Player Alive Status ---")
    alive_cols = [f"p{i}_alive" for i in range(6)]
    if not all(col in df.columns for col in alive_cols):
        logging.error("FAILED: Missing one or more player alive columns.")
        return False
    all_alive_values = pd.concat([df[col] for col in alive_cols]).unique()
    if any(v not in [0, 1] for v in all_alive_values):
        logging.error("FAILED: Invalid 'alive' status ID(s) found. Expected only 0 or 1.")
        return False
    logging.info("PASSED: All player 'alive' statuses are valid.")
    return True

def print_statistical_summary(df: pd.DataFrame, title: str = "Dataset Statistical Summary") -> None:
    """Prints a statistical summary of the given DataFrame with a custom title."""
    if df is None or df.empty:
        logging.warning(f"DataFrame for '{title}' is empty, cannot generate summary.")
        return
        
    logging.info(f"\n--- {title} ---") # <-- Use the new title parameter
    
    total_rows = len(df)
    blue_pos_samples = df['team_0_goal_in_event_window'].sum()
    orange_pos_samples = df['team_1_goal_in_event_window'].sum()
    
    # Calculate weights for each team separately
    blue_neg_samples = total_rows - blue_pos_samples
    orange_neg_samples = total_rows - orange_pos_samples
    
    blue_imbalance = blue_neg_samples / blue_pos_samples if blue_pos_samples > 0 else float('inf')
    orange_imbalance = orange_neg_samples / orange_pos_samples if orange_pos_samples > 0 else float('inf')
    
    logging.info(f"Total Rows (Game States): {total_rows:,}")
    logging.info(f"Unique Replays Found: {df['replay_id'].nunique()}")
    
    logging.info("\n-- Blue Team Stats --")
    logging.info(f"  Positive Samples (Blue Goal Window): {blue_pos_samples:,}")
    logging.info(f"  Class Imbalance (Negative / Positive): {blue_imbalance:.2f} : 1")
    
    logging.info("\n-- Orange Team Stats --")
    logging.info(f"  Positive Samples (Orange Goal Window): {orange_pos_samples:,}")
    logging.info(f"  Class Imbalance (Negative / Positive): {orange_imbalance:.2f} : 1")
    
    logging.info("--- End of Summary ---")

def validate_labeling_logic(df: pd.DataFrame, time_window: float, hertz: int) -> list[str]:
    """
    Performs a final, definitive, state-machine validation of goal labeling logic.
    Instead of failing, it now collects and returns a list of faulty replay_ids.

    Returns:
        A list of replay_ids that have inconsistencies. An empty list means success.
    """
    if df is None or df.empty: return []
    logging.info("--- Validating Goal Labeling Logic (Reporting Mode) ---")
    
    error_margin = 1.0 / hertz
    max_allowed_duration = time_window + error_margin
    logging.info(f"Max allowed label block duration: {max_allowed_duration:.3f}s.")

    required_cols_set = {'replay_id', 'time', 'blue_score', 'orange_score', 
                         'team_0_goal_in_event_window', 'team_1_goal_in_event_window'}
    if not required_cols_set.issubset(df.columns):
        logging.error("CRITICAL: Missing required columns for label validation. Cannot proceed.")
        # Return all replay IDs as faulty since we can't check any of them.
        return df['replay_id'].unique().tolist()

    faulty_replays = set()
    data_tuples = list(df[list(required_cols_set)].itertuples(index=True))

    in_window_for_team = -1
    window_start_time = 0.0
    window_start_score = 0
    last_replay_id = None

    # Add a dummy row to ensure the last real replay is fully processed
    dummy_row_data = {col: None for col in required_cols_set}
    dummy_row_data['replay_id'] = 'DUMMY_REPLAY_ID_FOR_FLUSH'
    dummy_row = pd.Series(dummy_row_data).to_frame().T.itertuples(index=True).__next__()
    data_tuples.append(dummy_row)

    for i in range(len(data_tuples)):
        row = data_tuples[i]
        
        if row.replay_id != last_replay_id:
            if row.replay_id == 'DUMMY_REPLAY_ID_FOR_FLUSH': break
            last_replay_id = row.replay_id
            in_window_for_team = -1
            logging.info(f"--- Analyzing labels for replay_id: {row.replay_id} ---")

        # Skip this iteration if the replay has already been marked as faulty
        if row.replay_id in faulty_replays:
            continue

        is_blue_labeled = row.team_0_goal_in_event_window == 1
        is_orange_labeled = row.team_1_goal_in_event_window == 1
        
        # (The state machine logic is identical to the previous version)
        if in_window_for_team == -1:
            if is_blue_labeled:
                in_window_for_team = 0
                window_start_time = row.time
                window_start_score = row.blue_score
            elif is_orange_labeled:
                in_window_for_team = 1
                window_start_time = row.time
                window_start_score = row.orange_score
        elif in_window_for_team == 0:
            score_changed = row.blue_score != window_start_score
            window_ended = not is_blue_labeled
            if window_ended or score_changed:
                if row.blue_score != window_start_score + 1:
                    logging.error(f"Replay {row.replay_id}: FAILED - Blue label window ending at index {data_tuples[i-1].Index} was not followed by a valid score increase.")
                    faulty_replays.add(row.replay_id)
                in_window_for_team = -1
                if is_blue_labeled and score_changed:
                    in_window_for_team = 0
                    window_start_time = row.time
                    window_start_score = row.blue_score
                elif is_orange_labeled:
                    in_window_for_team = 1
                    window_start_time = row.time
                    window_start_score = row.orange_score
            else:
                if row.time - window_start_time > max_allowed_duration:
                    logging.error(f"Replay {row.replay_id}: FAILED DURATION - Blue label window is too long.")
                    faulty_replays.add(row.replay_id)
        elif in_window_for_team == 1:
            score_changed = row.orange_score != window_start_score
            window_ended = not is_orange_labeled
            if window_ended or score_changed:
                if row.orange_score != window_start_score + 1:
                    logging.error(f"Replay {row.replay_id}: FAILED - Orange label window ending at index {data_tuples[i-1].Index} was not followed by a valid score increase.")
                    faulty_replays.add(row.replay_id)
                in_window_for_team = -1
                if is_orange_labeled and score_changed:
                    in_window_for_team = 1
                    window_start_time = row.time
                    window_start_score = row.orange_score
                elif is_blue_labeled:
                    in_window_for_team = 0
                    window_start_time = row.time
                    window_start_score = row.blue_score
            else:
                if row.time - window_start_time > max_allowed_duration:
                    logging.error(f"Replay {row.replay_id}: FAILED DURATION - Orange label window is too long.")
                    faulty_replays.add(row.replay_id)

    if not faulty_replays:
        logging.info("PASSED: Goal labeling logic is consistent and correct across all replays.")
    else:
        logging.error(f"FAILED: Found inconsistencies in {len(faulty_replays)} replay(s): {sorted(list(faulty_replays))}")
    
    return sorted(list(faulty_replays))

# ==================================================================
# Main Execution
# ==================================================================
# ====================== MAIN EXECUTION ======================
def main():
    """
    Main execution function for the dataset validation script.
    """
    start_time = time.monotonic()
    configure_logging()
    logging.info("Starting Dataset Validation Script for a SPLIT dataset folder...")

    validation_results = {}

    # --- Step 1: Find and load all CSVs ---
    if not DATASET_FOLDER_PATH.exists() or not DATASET_FOLDER_PATH.is_dir():
        logging.critical(f"Halting: Root dataset folder not found: {DATASET_FOLDER_PATH}")
        sys.exit(1)
    
    splits_to_validate = ['train', 'val', 'test']
    all_dfs = {}
    
    for split in splits_to_validate:
        split_path = DATASET_FOLDER_PATH / split
        if not split_path.is_dir():
            logging.warning(f"--- Split folder '{split}' not found. Skipping. ---"); continue
        
        csv_files = sorted(list(split_path.glob("*.csv")))
        if not csv_files:
            logging.warning(f"--- No CSV files in split folder '{split}'. Skipping. ---"); continue
            
        logging.info(f"--- Loading and concatenating {len(csv_files)} CSV chunks for '{split}' split... ---")
        try:
            # Load with the dtype hint, as you correctly did.
            list_of_dfs = [pd.read_csv(file, dtype={'replay_id': 'category'}) for file in csv_files]
            df = pd.concat(list_of_dfs, ignore_index=True)

            # --- THIS IS THE DEFINITIVE FIX ---
            # Force all values in the column to be strings, then convert to category.
            # This handles any hidden mixed types (e.g., numbers, NaNs) from the source files.
            df['replay_id'] = df['replay_id'].astype(str).astype('category')
            
            all_dfs[split] = df
            logging.info(f"Successfully loaded '{split}' split. Final shape: {df.shape}")
        except Exception as e:
            logging.error(f"Failed to load data for '{split}' split. Error: {e}")

    if not all_dfs:
        logging.critical("Halting: No valid data could be loaded from any split folder.")
        sys.exit(1)
    
    # Combine all loaded splits into one big DataFrame for overall validation checks
    full_combined_df = pd.concat(all_dfs.values(), ignore_index=True)
    # Re-apply the fix to the fully combined DataFrame as well
    full_combined_df['replay_id'] = full_combined_df['replay_id'].astype(str).astype('category')
    validation_results['File Load'] = True

    # --- Step 2: Run all validation checks on the complete, combined DataFrame ---
    checks_to_run = [
        ("Schema", validate_schema, (full_combined_df, EXPECTED_SCHEMA)),
        ("Missing Values (NaN)", validate_missing_values, (full_combined_df,)),
        ("Infinity Values", validate_infinity_values, (full_combined_df,)),
        ("Boost Amounts", validate_boost_amount, (full_combined_df,)),
        ("Team Composition", validate_team_composition, (full_combined_df,)),
        ("Player Alive Status", validate_player_alive_status, (full_combined_df,)),
    ]

    for name, func, args in checks_to_run:
        validation_results[name] = func(*args)
        if name == "Schema" and not validation_results[name]:
            logging.critical("Halting due to critical schema failure.")
            print_final_summary(validation_results, time.monotonic() - start_time)
            sys.exit(1)

    validation_results['Playable Area'] = validate_playable_area(full_combined_df)
    faulty_replay_ids = validate_labeling_logic(full_combined_df, GOAL_ANTICIPATION_WINDOW_SECONDS, DATASET_HERTZ)
    validation_results['Goal Labeling Logic'] = not bool(faulty_replay_ids)

    # --- Step 3: Print the Statistical Summary for each split individually ---
    for split_name, df_split in all_dfs.items():
        print_statistical_summary(df_split, title=f"'{split_name.upper()}' Split Statistical Summary")

    # Optionally, print the summary for the entire dataset as well
    print_statistical_summary(full_combined_df, title="OVERALL Dataset Statistical Summary")
    
    # --- Step 4: Final Report ---
    total_time = time.monotonic() - start_time
    print_final_summary(validation_results, total_time)
    
    if faulty_replay_ids:
        logging.warning("\n" + "="*50)
        logging.warning("ACTION REQUIRED: The following faulty replay IDs were found and should be removed from the dataset:")
        for replay_id in faulty_replay_ids:
            logging.warning(f"  - {replay_id}")
        logging.warning("="*50)
    
    if not all(validation_results.values()):
        logging.error("One or more validation checks failed.")
    else:
        logging.info("All validation checks passed successfully!")

if __name__ == "__main__":
    main()