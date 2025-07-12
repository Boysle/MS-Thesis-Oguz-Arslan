import pandas as pd
from pathlib import Path
import logging
import shutil

# ==================================================================
# Configuration
# ==================================================================
# The folder containing the dataset chunks you want to clean
DATASET_FOLDER = Path(r"E:\\Raw RL Esports Replays\\Big Replay Dataset\\chunked_dataset")

# The list of faulty replay IDs you got from the validation script
FAULTY_REPLAY_IDS = [
    "replay_00164",
    # Add any other faulty replay IDs here
]

# Where to save the new, cleaned files
CLEANED_OUTPUT_FOLDER = DATASET_FOLDER.parent / "dataset_cleaned_v2"

LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# ==================================================================
# Script Logic
# ==================================================================
def clean_dataset_efficiently(source_folder: Path, dest_folder: Path, ids_to_remove: list[str]):
    """
    Efficiently cleans a chunked dataset by removing all data associated
    with specified faulty replay IDs.
    """
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, handlers=[
        logging.FileHandler("dataset_cleaning.log", mode='w'),
        logging.StreamHandler()
    ])
    
    if not source_folder.is_dir():
        logging.error(f"Source folder not found: {source_folder}")
        return
        
    dest_folder.mkdir(parents=True, exist_ok=True)
    logging.info(f"Cleaned files will be saved in: {dest_folder}")
    
    csv_files = sorted(list(source_folder.glob("*.csv")))
    if not csv_files:
        logging.error(f"No CSV files found in {source_folder}")
        return
        
    # Use a set for efficient lookup of faulty IDs
    faulty_ids_set = set(ids_to_remove)
    total_rows_before = 0
    total_rows_after = 0
    
    logging.info(f"Starting cleaning process. Will remove {len(faulty_ids_set)} faulty replay(s)...")

    for file_path in csv_files:
        logging.info(f"Processing chunk: {file_path.name}")
        output_path = dest_folder / file_path.name

        try:
            # First, quickly check which replays are in this chunk without loading the whole file.
            # We only need to load the 'replay_id' column.
            chunk_replay_ids = pd.read_csv(file_path, usecols=['replay_id'], dtype={'replay_id': 'category'})['replay_id'].unique()
            
            # Find which faulty replays, if any, are present in this specific chunk.
            ids_in_this_chunk_to_remove = faulty_ids_set.intersection(chunk_replay_ids)

            if not ids_in_this_chunk_to_remove:
                # --- OPTIMIZATION ---
                # This chunk is clean! No need to process it. Just copy the file.
                logging.info("  -> Chunk is clean. Copying file directly.")
                shutil.copy(file_path, output_path)
                # We still need the row count for the final summary
                row_count = len(chunk_replay_ids) # This is an approximation but fast
                # For exact count, we'd need to read the whole file again.
                # Let's do it properly.
                df_temp = pd.read_csv(file_path, low_memory=False)
                row_count = len(df_temp)
                del df_temp

                total_rows_before += row_count
                total_rows_after += row_count
                continue

            # This chunk contains faulty data and needs to be processed.
            logging.warning(f"  -> Found faulty replays in this chunk: {ids_in_this_chunk_to_remove}. Cleaning required.")
            
            # Now load the full chunk for processing
            df = pd.read_csv(file_path, dtype={'replay_id': 'category'})
            
            rows_before = len(df)
            total_rows_before += rows_before
            
            # The cleaning logic is a single, powerful line of code
            df_cleaned = df[~df['replay_id'].isin(ids_in_this_chunk_to_remove)]
            
            rows_after = len(df_cleaned)
            total_rows_after += rows_after
            
            rows_removed = rows_before - rows_after
            logging.info(f"  -> Removed {rows_removed} rows. Saving cleaned chunk.")
                
            # Save the cleaned chunk
            df_cleaned.to_csv(output_path, index=False)

        except Exception as e:
            logging.error(f"Failed to process chunk {file_path.name}. Error: {e}. Copying original file as a fallback.")
            # As a safeguard, copy the original file so the dataset isn't missing a chunk.
            shutil.copy(file_path, output_path)


    logging.info("--- Cleaning Complete ---")
    logging.info(f"Total rows before cleaning: {total_rows_before:,}")
    logging.info(f"Total rows after cleaning:  {total_rows_after:,}")
    logging.info(f"Total rows removed:         {(total_rows_before - total_rows_after):,}")

if __name__ == "__main__":
    clean_dataset_efficiently(DATASET_FOLDER, CLEANED_OUTPUT_FOLDER, FAULTY_REPLAY_IDS)