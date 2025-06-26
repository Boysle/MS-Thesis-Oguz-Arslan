import pandas as pd
import logging
import numpy as np
from pathlib import Path

# ==================================================================
# Configuration
# ==================================================================
# Path to the dataset CSV file
# Update this path if your actual filename is different (e.g., has .csv extension)
# Assuming the full filename is "dataset_5hz_5sec_Test_Sample_Replays.csv" based on your previous naming
DATASET_PATH = Path(r"D:\\Raw RL Esports Replays\\Test Sample Replays\\dataset_5hz_5sec_Test_Sample_Replays.csv")

LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# Define the expected schema (column name -> expected dtype string)
# Using strings for dtype comparison is generally robust enough for pandas dtypes.
EXPECTED_SCHEMA = {
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
    'ball_hit_team_num': 'float64', # Float is correct here since if no hit yet, value is 0.5
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

# The EXPECTED_COLUMNS list can now be derived from EXPECTED_SCHEMA keys
EXPECTED_COLUMNS = list(EXPECTED_SCHEMA.keys())


# ==================================================================
# Logging Configuration
# ==================================================================
def configure_logging():
    """Set up logging."""
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, handlers=[
        logging.FileHandler("dataset_validation.log", mode='w'), # Overwrite log file each run
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
        df = pd.read_csv(file_path)
        logging.info(f"Dataset loaded successfully. Shape: {df.shape}")
        if df.empty:
            logging.warning("Loaded dataset is empty.")
        return df
    except FileNotFoundError:
        # This case should be caught by validate_file_existence, but good to have defense
        logging.error(f"Error loading dataset: File not found at {file_path}")
        return None
    except pd.errors.EmptyDataError:
        logging.error(f"Error loading dataset: File is empty at {file_path}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading dataset: {e}")
        return None
    
def validate_schema(df: pd.DataFrame, expected_schema: dict) -> bool:
    """
    Validates column presence, naming, and data types against an expected schema.
    """
    if df is None:
        logging.error("DataFrame is None, cannot validate schema.")
        return False

    logging.info("--- Validating Dataset Schema (Columns and Data Types) ---")
    overall_success = True

    # 1. Check for missing expected columns
    actual_cols_set = set(df.columns)
    expected_cols_set = set(expected_schema.keys())
    missing_cols = expected_cols_set - actual_cols_set
    if missing_cols:
        logging.error(f"MISSING expected columns: {sorted(list(missing_cols))}")
        overall_success = False
    else:
        logging.info("All expected columns are present.")

    # 2. Check for unexpected (extra) columns
    extra_cols = actual_cols_set - expected_cols_set
    if extra_cols:
        logging.warning(f"Found UNEXPECTED columns: {sorted(list(extra_cols))}")
        # Not necessarily a failure, but good to flag.

    # 3. Check data types for present expected columns
    logging.info("Checking data types for expected columns:")
    type_mismatches = {}
    for col_name, expected_dtype_str in expected_schema.items():
        if col_name in df.columns:
            actual_dtype_str = str(df[col_name].dtype)
            if actual_dtype_str != expected_dtype_str:
                type_mismatches[col_name] = {'expected': expected_dtype_str, 'actual': actual_dtype_str}
                logging.error(f"Column '{col_name}': DType Mismatch! Expected: {expected_dtype_str}, Actual: {actual_dtype_str}")
                overall_success = False
            # else:
                # logging.debug(f"Column '{col_name}': DType OK ({actual_dtype_str})")
        # else: column is missing, already logged above

    if not type_mismatches:
        logging.info("All present expected columns have correct data types.")
    
    if overall_success:
        logging.info("Schema validation PASSED.")
    else:
        logging.error("Schema validation FAILED.")
    
    logging.info("--- End of Schema Validation ---")
    return overall_success    

def validate_missing_values(df: pd.DataFrame, log_nan_indices: bool = False, max_indices_to_log: int = 10) -> bool:
    """
    Checks for missing (NaN) values in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to validate.
        log_nan_indices (bool): If True, logs the row indices of NaNs for each affected column.
                                Can be verbose for large datasets with many NaNs.
        max_indices_to_log (int): The maximum number of NaN indices to log per column if log_nan_indices is True.

    Returns:
        bool: True if no NaNs are found, False otherwise.
    """
    if df is None:
        logging.error("DataFrame is None, cannot validate missing values.")
        return False # Or raise an error, depending on desired strictness

    logging.info("--- Validating Missing Values (NaNs) ---")
    overall_success = True
    any_nans_found_in_dataset = False

    nan_summary = df.isnull().sum()
    columns_with_nans = nan_summary[nan_summary > 0]

    if columns_with_nans.empty:
        logging.info("PASSED: No NaN values found in any column.")
    else:
        logging.error("FAILED: NaN values found in the dataset!")
        any_nans_found_in_dataset = True
        overall_success = False
        logging.error("Summary of columns with NaNs:")
        for col_name, nan_count in columns_with_nans.items():
            nan_percentage = (nan_count / len(df)) * 100
            logging.error(
                f"  Column '{col_name}': {nan_count} NaNs ({nan_percentage:.2f}%)"
            )
            if log_nan_indices:
                nan_indices = df[df[col_name].isnull()].index.tolist()
                if nan_indices:
                    indices_to_show = nan_indices[:max_indices_to_log]
                    logging.error(f"    NaNs at (up to {max_indices_to_log}) row indices: {indices_to_show}")
                    if len(nan_indices) > max_indices_to_log:
                        logging.error(f"    ... and {len(nan_indices) - max_indices_to_log} more NaN indices not shown.")

    if not any_nans_found_in_dataset:
        # This double check is a bit redundant if columns_with_nans.empty was true,
        # but confirms the overall_success logic.
        logging.info("Final check: No NaNs detected across the dataset.")
    
    logging.info("--- End of Missing Values Validation ---")
    return overall_success

# ==================================================================
# Main Execution
def main():
    configure_logging() # Ensure this is called
    logging.info("Starting Dataset Validation Script...")

    if not validate_file_existence(DATASET_PATH):
        logging.critical("Halting validation: Dataset file not found or inaccessible.")
        return

    df = load_dataset(DATASET_PATH)
    if df is None:
        logging.critical("Halting validation: Failed to load dataset.")
        return
    if df.empty:
        logging.warning("Dataset is empty. Some checks might not be meaningful.")
        # Potentially return here if an empty df is a critical failure for subsequent checks

    # Validate schema (covers column presence and data types)
    if not validate_schema(df, EXPECTED_SCHEMA):
        logging.critical("Halting validation: Core schema validation failed.")
        return # It's generally good to halt if the basic structure is wrong
        
    # Validate Missing Values
    # Set log_nan_indices=True to see specific row indices (can be very verbose)
    if not validate_missing_values(df, log_nan_indices=True, max_indices_to_log=5):
        # Decide if this is a critical failure that should halt further validation
        logging.error("Missing values found! This is unexpected after data processing.")
        # You might want to halt here:
        # logging.critical("Halting validation: Missing values detected.")
        # return
    else:
        logging.info("Missing values check passed (or no new NaNs found).")

    # Example of calling another specific validation if you had one:
    # validate_ball_hit_team_num(df) 

    logging.info("Dataset Validation Script Finished.")

if __name__ == "__main__":
    main()