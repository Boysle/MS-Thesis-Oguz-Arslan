import pandas as pd
import logging
import numpy as np
from pathlib import Path

# ==================================================================
# Configuration
# ==================================================================
# Path to the dataset CSV file
DATASET_PATH = Path(r"D:\\Raw RL Esports Replays\\Test Sample Replays\\dataset_5hz_5sec_Test_Sample_Replays.csv")

LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# How many seconds before a goal is the event window? Must match the converter script.
GOAL_ANTICIPATION_WINDOW_SECONDS = 5.0 

# Define the expected schema (column name -> expected dtype string)
EXPECTED_SCHEMA = {
    # --- New Context Columns ---
    'replay_id': 'category',
    'blue_score': 'int64',
    'orange_score': 'int64',
    'score_difference': 'int64',
    
    # --- Existing Columns ---
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
    'p3_dist_to_ball': 'float64', 'p4_dist_to_ball': 'float64', 'p5_dist_to_ball': 'float64'
}

# Field boundaries for coordinate validation
FIELD_BOUNDS = {
    'x': (-4096, 4096),
    'y': (-6000, 6000),
    'z': (0, 2044)
}

# ==================================================================
# Logging Configuration
# ==================================================================
def configure_logging():
    """Set up logging."""
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, handlers=[
        logging.FileHandler("dataset_validation_results.log", mode='w'), # Overwrite log file each run
        logging.StreamHandler()
    ])

# ==================================================================
# Validation Functions
# ==================================================================

def validate_file_existence(file_path: Path) -> bool:
    """Checks if the dataset file exists."""
    logging.info(f"Checking existence of dataset file: {file_path}")
    if not file_path.exists():
        logging.error(f"Dataset file NOT FOUND: {file_path}")
        return False
    if not file_path.is_file():
        logging.error(f"Path exists but is NOT A FILE: {file_path}")
        return False
    logging.info("Dataset file found.")
    return True

def load_dataset(file_path: Path) -> pd.DataFrame | None:
    """Loads the dataset from CSV."""
    logging.info(f"Attempting to load dataset from: {file_path}")
    try:
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
    logging.info("--- Validating Dataset Schema (Columns and Data Types) ---")
    overall_success = True
    actual_cols_set = set(df.columns)
    expected_cols_set = set(expected_schema.keys())
    
    missing_cols = expected_cols_set - actual_cols_set
    if missing_cols:
        logging.error(f"MISSING expected columns: {sorted(list(missing_cols))}")
        overall_success = False
    
    extra_cols = actual_cols_set - expected_cols_set
    if extra_cols:
        logging.warning(f"Found UNEXPECTED columns: {sorted(list(extra_cols))}")

    for col_name, expected_dtype in expected_schema.items():
        if col_name in df.columns:
            actual_dtype = str(df[col_name].dtype)
            if actual_dtype != expected_dtype:
                logging.error(f"Column '{col_name}': DType Mismatch! Expected: {expected_dtype}, Actual: {actual_dtype}")
                overall_success = False
    
    if overall_success:
        logging.info("Schema validation PASSED.")
    else:
        logging.error("Schema validation FAILED.")
    return overall_success    

def validate_missing_values(df: pd.DataFrame) -> bool:
    """Checks for any NaN values in the DataFrame."""
    if df is None: return False
    logging.info("--- Validating Missing Values (NaNs) ---")
    nan_summary = df.isnull().sum()
    columns_with_nans = nan_summary[nan_summary > 0]
    if columns_with_nans.empty:
        logging.info("PASSED: No NaN values found in any column.")
        return True
    else:
        logging.error("FAILED: NaN values found in the dataset!")
        for col, count in columns_with_nans.items():
            logging.error(f"  - Column '{col}': {count} NaNs")
        return False

def validate_infinity_values(df: pd.DataFrame) -> bool:
    """Checks for any infinity values in the DataFrame."""
    if df is None: return False
    logging.info("--- Validating Infinity Values (inf, -inf) ---")
    numeric_df = df.select_dtypes(include=np.number)
    is_inf_df = numeric_df.apply(np.isinf)
    inf_summary = is_inf_df.sum()
    columns_with_inf = inf_summary[inf_summary > 0]
    if columns_with_inf.empty:
        logging.info("PASSED: No infinity values found in any column.")
        return True
    else:
        logging.error("FAILED: Infinity values found in the dataset!")
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
        logging.error(f"FAILED: Invalid team ID(s) found. Expected only 0 or 1.")
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
        logging.error(f"FAILED: Invalid 'alive' status ID(s) found. Expected only 0 or 1.")
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
    logging.info(f"Positive Samples (Blue Goal Window): {blue_pos_samples:,}")
    logging.info(f"Positive Samples (Orange Goal Window): {orange_pos_samples:,}")
    logging.info(f"Total Positive Samples: {total_pos_samples:,}")
    logging.info(f"Total Negative Samples: {total_neg_samples:,}")
    logging.info(f"Class Imbalance (Negative / Positive): {imbalance_ratio:.2f} : 1")
    logging.info("--- End of Statistical Summary ---")

def validate_labeling_logic(df: pd.DataFrame, time_window: float) -> bool:
    """
    Performs a robust, in-depth validation of the goal labeling logic by
    ensuring every positive label is "justified" by a subsequent goal event.

    This is the definitive check for "orphan" labels.
    """
    if df is None or df.empty: return False
    logging.info("--- Validating Goal Labeling Logic (Orphan Label Check) ---")
    overall_success = True
    
    required_cols = {'replay_id', 'time', 'blue_score', 'orange_score', 
                     'team_0_goal_in_event_window', 'team_1_goal_in_event_window'}
    if not required_cols.issubset(df.columns):
        logging.error(f"FAILED: Missing one or more required columns for label validation: {required_cols - set(df.columns)}")
        return False

    # Group by replay to handle events in their own context
    for replay_id, replay_df in df.groupby('replay_id'):
        logging.info(f"--- Analyzing labels for replay_id: {replay_id} ---")
        replay_success = True
        
        replay_df = replay_df.copy()

        # Identify all goal events for this replay for efficient lookup
        replay_df['blue_score_change'] = replay_df['blue_score'].diff().fillna(0)
        replay_df['orange_score_change'] = replay_df['orange_score'].diff().fillna(0)
        
        blue_goal_times = replay_df.loc[replay_df['blue_score_change'] > 0, 'time'].tolist()
        orange_goal_times = replay_df.loc[replay_df['orange_score_change'] > 0, 'time'].tolist()

        for team_num, goal_times, label_col in [(0, blue_goal_times, 'team_0_goal_in_event_window'), 
                                                (1, orange_goal_times, 'team_1_goal_in_event_window')]:
            
            # Find all rows that are positively labeled for this team
            labeled_rows = replay_df[replay_df[label_col] == 1]
            
            if labeled_rows.empty:
                continue # No labels to check for this team

            # For each labeled row, ensure a goal follows within the time window
            for label_idx, labeled_row in labeled_rows.iterrows():
                label_time = labeled_row['time']
                
                # A label at `label_time` is justified if a goal exists at `goal_time`
                # where `label_time <= goal_time <= label_time + 5s`.
                is_justified = any(
                    label_time <= goal_time <= label_time + time_window
                    for goal_time in goal_times
                )
                
                if not is_justified:
                    replay_success = False
                    logging.error(f"Replay {replay_id}: FAILED - Found 'orphan' positive label for Team {team_num} at index {label_idx} (time {label_time:.2f}). No corresponding goal was found within the next {time_window} seconds.")
                    # Break after the first orphan in this replay to avoid spamming logs
                    break
            
            if not replay_success:
                break # Stop checking this replay if an error was found

        if replay_success:
            logging.info(f"Replay {replay_id}: PASSED - All positive labels are correctly associated with a goal.")
        else:
            overall_success = False

    if overall_success:
        logging.info("PASSED: Goal labeling logic is consistent and correct across all replays.")
    else:
        logging.error("FAILED: Found one or more 'orphan' labels. See details above.")

    return overall_success

# ==================================================================
# Main Execution
# ==================================================================
def main():
    configure_logging()
    logging.info("Starting Dataset Validation Script...")

    if not validate_file_existence(DATASET_PATH):
        logging.critical("Halting validation: Dataset file not found.")
        return

    df = load_dataset(DATASET_PATH)
    if df is None or df.empty:
        logging.critical("Halting validation: Failed to load or empty dataset.")
        return

    # Create a list of all validation functions to run
    validation_checks = [
        (validate_schema, (df, EXPECTED_SCHEMA)),
        (validate_missing_values, (df,)),
        (validate_infinity_values, (df,)),
        (validate_coordinate_ranges, (df, FIELD_BOUNDS)),
        (validate_boost_amount, (df,)),
        (validate_team_composition, (df,)),
        (validate_player_alive_status, (df,)),
    ]

    all_checks_passed = True
    for func, args in validation_checks:
        if not func(*args):
            all_checks_passed = False

    if not all_checks_passed:
        logging.error("One or more basic data integrity checks failed. Label validation might be unreliable.")
    
    # --- STATISTICAL AND LABELING VALIDATIONS ---
    # These are run regardless of previous checks to provide insight.
    print_statistical_summary(df)
    validate_labeling_logic(df, GOAL_ANTICIPATION_WINDOW_SECONDS)

    logging.info("Dataset Validation Script Finished.")

if __name__ == "__main__":
    main()