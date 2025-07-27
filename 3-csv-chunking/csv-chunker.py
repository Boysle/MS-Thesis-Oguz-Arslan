import pandas as pd
from pathlib import Path
import logging
import math

# ==================================================================
# Configuration
# ==================================================================
# --- INPUT ---
# Path to our massive dataset file
LARGE_CSV_PATH = Path(r"E:\\Raw RL Esports Replays\\Day 3 Swiss Stage\\Round 1\\dataset_5hz_5sec_Round_1.csv")

# --- OUTPUT ---
# The folder where you want to save the smaller chunked files
OUTPUT_FOLDER = LARGE_CSV_PATH.parent / "chunked_dataset"

# --- SETTINGS ---
# The desired number of output files
NUMBER_OF_CHUNKS = 10
LOG_LEVEL = logging.INFO

# ==================================================================
# Script Logic
# ==================================================================
def chunk_large_csv(source_path: Path, dest_folder: Path, num_chunks: int):
    """
    Reads a large CSV file and splits it into a specified number of smaller CSV files.

    Args:
        source_path: The path to the large input CSV file.
        dest_folder: The directory where the chunked files will be saved.
        num_chunks: The desired number of smaller output files.
    """
    logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if not source_path.exists():
        logging.error(f"Source file not found: {source_path}")
        return

    # Create the destination folder if it doesn't exist
    dest_folder.mkdir(parents=True, exist_ok=True)
    logging.info(f"Chunked files will be saved in: {dest_folder}")

    try:
        # First, get the total number of rows to calculate chunk sizes
        logging.info("Counting total rows in the source file (this may take a moment)...")
        # This is a memory-efficient way to count rows
        total_rows = sum(1 for row in open(source_path, 'r', encoding='utf-8')) - 1 # Subtract 1 for the header
        logging.info(f"Source file has {total_rows:,} rows.")

        if total_rows == 0:
            logging.warning("Source file is empty. No chunks will be created.")
            return

        # Calculate the number of rows per chunk file
        chunk_size = math.ceil(total_rows / num_chunks)
        logging.info(f"Splitting into {num_chunks} files with approximately {chunk_size:,} rows each.")

        # Use pandas to read the CSV in chunks and write to new files
        # The 'chunksize' parameter is key here - it prevents loading the whole file into memory
        reader = pd.read_csv(source_path, chunksize=chunk_size)
        
        for i, df_chunk in enumerate(reader):
            chunk_number = i + 1
            output_filename = dest_folder / f"dataset_chunk_{chunk_number:02d}.csv"
            logging.info(f"Writing chunk {chunk_number}/{num_chunks} to {output_filename.name}...")
            
            # Write the chunk to a new CSV file. index=False is important.
            df_chunk.to_csv(output_filename, index=False)
            
        logging.info("Successfully split the large CSV into smaller chunks.")

    except Exception as e:
        logging.error(f"An error occurred during the chunking process: {e}")

if __name__ == "__main__":
    chunk_large_csv(LARGE_CSV_PATH, OUTPUT_FOLDER, NUMBER_OF_CHUNKS)