import pandas as pd
import logging
import numpy as np
from pathlib import Path
import time
import sys

# ==================================================================
# Configuration
# ==================================================================
# Path to the dataset CSV file
DATASET_PATH = Path(r"E:\\Raw RL Esports Replays\\Big Replay Dataset\\dataset_5hz_5sec_Big_Replay_Dataset.csv")

# --- Validation Parameters ---
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
GOAL_ANTICIPATION_WINDOW_SECONDS = 5.0
DATASET_HERTZ = 5 # IMPORTANT: Set this to match the HZ of the dataset file
FIELD_BOUNDS = {'x': (-4096, 4096), 'y': (-6000, 6000), 'z': (0, 2044)}

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
    """Set up logging."""
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, handlers=[
        logging.FileHandler("dataset_validation_results.log", mode='w'),
        logging.StreamHandler()
    ])

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
    
    # Log the summary to file and console
    for line in summary_lines:
        logging.info(line)

# ==================================================================
# Validation Functions
# ==================================================================

def validate_file_existence(file_path: Path) -> bool:
    """Checks if the dataset file exists."""
    logging.info("--- Validating File Existence ---")
    if not file_path.exists():
        logging.error(f"FAILED: Dataset file NOT FOUND: {file_path}")
        return False
    logging.info("PASSED: Dataset file found.")
    return True

def load_dataset(file_path: Path) -> pd.DataFrame | None:
    """Loads the dataset from CSV."""
    logging.info(f"Attempting to load dataset from: {file_path}")
    try:
        # Explicitly set dtype for replay_id during load for efficiency
        df = pd.read_csv(file_path, dtype={'replay_id': 'category'})
        logging.info(f"Dataset loaded successfully. Shape: {df.shape}")
        if df.empty:
            logging.warning("Loaded dataset is empty.")
        return df
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading dataset: {e}")
        return None
    
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

def validate_coordinate_ranges(df: pd.DataFrame, bounds: dict) -> bool:
    """Checks if player and ball coordinates are within expected field boundaries."""
    if df is None: return False
    logging.info("--- Validating Coordinate Ranges ---")
    overall_success = True
    subjects = [f'p{i}' for i in range(6)] + ['ball']
    for subject in subjects:
        for axis, (min_val, max_val) in bounds.items():
            col_name = f"{subject}_pos_{axis}"
            if col_name in df.columns:
                outliers = df[(df[col_name] < min_val) | (df[col_name] > max_val)]
                if not outliers.empty:
                    overall_success = False
                    logging.error(f"FAILED: Column '{col_name}' has {len(outliers)} values outside range [{min_val}, {max_val}].")
    if overall_success:
        logging.info("PASSED: All coordinates are within defined limits.")
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

def print_statistical_summary(df: pd.DataFrame) -> None:
    """Prints a statistical summary of the dataset."""
    if df is None or df.empty:
        logging.warning("DataFrame is empty, cannot generate statistical summary.")
        return
        
    logging.info("--- Dataset Statistical Summary ---")
    total_rows = len(df)
    blue_pos_samples = df['team_0_goal_in_event_window'].sum()
    orange_pos_samples = df['team_1_goal_in_event_window'].sum()
    total_pos_samples = blue_pos_samples + orange_pos_samples
    total_neg_samples = total_rows - total_pos_samples
    imbalance_ratio = total_neg_samples / total_pos_samples if total_pos_samples > 0 else float('inf')
    
    logging.info(f"Total Rows (Game States): {total_rows:,}")
    logging.info(f"Unique Replays Found: {df['replay_id'].nunique()}")
    logging.info(f"Positive Samples (Blue Goal Window): {blue_pos_samples:,}")
    logging.info(f"Positive Samples (Orange Goal Window): {orange_pos_samples:,}")
    logging.info(f"Total Positive Samples: {total_pos_samples:,}")
    logging.info(f"Total Negative Samples: {total_neg_samples:,}")
    logging.info(f"Class Imbalance (Negative / Positive): {imbalance_ratio:.2f} : 1")
    logging.info("--- End of Statistical Summary ---")

def validate_labeling_logic(df: pd.DataFrame, time_window: float, hertz: int) -> bool:
    """
    Performs a final, definitive, state-machine validation of goal labeling logic.
    """
    if df is None or df.empty: return False
    logging.info("--- Validating Goal Labeling Logic ---")
    
    error_margin = 1.0 / hertz
    max_allowed_duration = time_window + error_margin
    logging.info(f"Max allowed label block duration: {max_allowed_duration:.3f}s.")

    required_cols_set = {'replay_id', 'time', 'blue_score', 'orange_score', 
                         'team_0_goal_in_event_window', 'team_1_goal_in_event_window'}
    if not required_cols_set.issubset(df.columns):
        logging.error("FAILED: Missing required columns for label validation.")
        return False

    required_cols_list = list(required_cols_set)
    data_tuples = list(df[required_cols_list].itertuples(index=True))

    in_window_for_team = -1
    window_start_time = 0.0
    window_start_score = 0
    last_replay_id = None
    overall_success = True

    for i in range(len(data_tuples)):
        row = data_tuples[i]
        
        if row.replay_id != last_replay_id:
            if in_window_for_team != -1:
                logging.debug(f"Label window for replay {last_replay_id} correctly ends at end of data.")
            last_replay_id = row.replay_id
            in_window_for_team = -1
            logging.info(f"--- Analyzing labels for replay_id: {row.replay_id} ---")

        is_blue_labeled = row.team_0_goal_in_event_window == 1
        is_orange_labeled = row.team_1_goal_in_event_window == 1
        
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
            if not is_blue_labeled:
                if row.blue_score != window_start_score + 1:
                    logging.error(f"Replay {row.replay_id}: FAILED ORPHAN - Blue label window ending at index {data_tuples[i-1].Index} was not followed by a score increase.")
                    overall_success = False; break
                in_window_for_team = -1
                if is_orange_labeled:
                    in_window_for_team = 1
                    window_start_time = row.time
                    window_start_score = row.orange_score
            else:
                duration = row.time - window_start_time
                if duration > max_allowed_duration:
                    logging.error(f"Replay {row.replay_id}: FAILED DURATION - Blue label window starting at {window_start_time:.2f}s is too long.")
                    overall_success = False; break
                
                if row.blue_score != window_start_score:
                    last_frame = data_tuples[i-1]
                    if row.blue_score != last_frame.blue_score + 1:
                         logging.error(f"Replay {row.replay_id}: FAILED CONSECUTIVE GOAL - Blue label window ending at index {last_frame.Index} was not followed by a valid score increase.")
                         overall_success = False; break
                    logging.debug(f"Replay {row.replay_id}: Detected consecutive Blue goal. Validating previous window and starting new one at index {row.Index}.")
                    window_start_time = row.time
                    window_start_score = row.blue_score
        
        elif in_window_for_team == 1:
            if not is_orange_labeled:
                if row.orange_score != window_start_score + 1:
                    logging.error(f"Replay {row.replay_id}: FAILED ORPHAN - Orange label window ending at index {data_tuples[i-1].Index} was not followed by a score increase.")
                    overall_success = False; break
                in_window_for_team = -1
                if is_blue_labeled:
                    in_window_for_team = 0
                    window_start_time = row.time
                    window_start_score = row.blue_score
            else:
                duration = row.time - window_start_time
                if duration > max_allowed_duration:
                    logging.error(f"Replay {row.replay_id}: FAILED DURATION - Orange label window starting at {window_start_time:.2f}s is too long.")
                    overall_success = False; break
                
                if row.orange_score != window_start_score:
                    last_frame = data_tuples[i-1]
                    if row.orange_score != last_frame.orange_score + 1:
                         logging.error(f"Replay {row.replay_id}: FAILED CONSECUTIVE GOAL - Orange label window ending at index {last_frame.Index} was not followed by a valid score increase.")
                         overall_success = False; break
                    logging.debug(f"Replay {row.replay_id}: Detected consecutive Orange goal. Validating previous window and starting new one at index {row.Index}.")
                    window_start_time = row.time
                    window_start_score = row.orange_score

        if not overall_success:
            break

    if overall_success:
        logging.info("PASSED: Goal labeling logic is consistent.")
    else:
        logging.error("FAILED: Found one or more inconsistencies in goal labeling logic.")
    return overall_success

# ==================================================================
# Main Execution
# ==================================================================
def main():
    start_time = time.monotonic()
    configure_logging()
    logging.info("Starting Dataset Validation Script...")

    validation_results = {}

    if not validate_file_existence(DATASET_PATH):
        validation_results['File Existence'] = False
        logging.critical("Halting: Dataset file not found.")
        print_final_summary(validation_results, time.monotonic() - start_time)
        sys.exit(1)
    validation_results['File Existence'] = True

    df = load_dataset(DATASET_PATH)
    if df is None or df.empty:
        validation_results['File Load'] = False
        logging.critical("Halting: Failed to load or empty dataset.")
        print_final_summary(validation_results, time.monotonic() - start_time)
        sys.exit(1)
    validation_results['File Load'] = True

    # Run checks in a logical order
    validation_results['Schema'] = validate_schema(df, EXPECTED_SCHEMA)
    if not validation_results['Schema']:
        logging.critical("Halting due to critical schema failure.")
        print_final_summary(validation_results, time.monotonic() - start_time)
        sys.exit(1)

    validation_results['Missing Values (NaN)'] = validate_missing_values(df)
    validation_results['Infinity Values'] = validate_infinity_values(df)
    validation_results['Coordinate Ranges'] = validate_coordinate_ranges(df, FIELD_BOUNDS)
    validation_results['Boost Amounts'] = validate_boost_amount(df)
    validation_results['Team Composition'] = validate_team_composition(df)
    validation_results['Player Alive Status'] = validate_player_alive_status(df)
    
    # Run the definitive labeling logic validation
    validation_results['Goal Labeling Logic'] = validate_labeling_logic(df, GOAL_ANTICIPATION_WINDOW_SECONDS, DATASET_HERTZ)

    # --- Informational Summaries ---
    # These don't have a pass/fail, they just provide analysis.
    print_statistical_summary(df)

    # --- Final Report ---
    total_time = time.monotonic() - start_time
    print_final_summary(validation_results, total_time)
    
    if not all(validation_results.values()):
        logging.error("One or more validation checks failed.")
        sys.exit(1)
    else:
        logging.info("All validation checks passed successfully!")

if __name__ == "__main__":
    main()