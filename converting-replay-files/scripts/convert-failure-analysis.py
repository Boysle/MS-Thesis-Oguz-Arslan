import argparse
from pathlib import Path
import re
import logging
import sys
import io
import time

# Configure basic logging for this analysis script itself
LOG_ANALYSIS_FILENAME = "failure_analysis_script.log"

def configure_analysis_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler(LOG_ANALYSIS_FILENAME, encoding='utf-8', mode='w'), # Overwrite for each analysis run
            logging.StreamHandler(sys.stdout)
        ]
    )

# Fix for Windows console encoding if needed for this script's output
if sys.stdout.encoding != 'UTF-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def find_replay_original_path_in_log(replay_name: str, log_lines: list[str], parent_search_dir: Path) -> str:
    """
    Tries to find the original full path of a replay.
    1. Searches log lines for when carball processing started for this replay.
    2. As a fallback, searches the parent_search_dir for the replay file.
    """
    # Pattern 1: "Processing [replay_name]..." might be logged by your main script
    # Pattern 2: "run_carball: Processing [full_path_to_replay_name]" - if run_carball logged this
    # Pattern 3: Log lines from find_replay_files (if it logged found paths)
    # Most reliable: The log line from your main processing loop:
    # "INFO - replay_converter:XXX - --- [X/Y] Checking result for: [replay_name] ---"
    # This doesn't give the path.
    # The line "INFO - replay_converter:XXX - Processing [replay_name]..." is better.

    # More robust: Carball log line "INFO - carball.rattletrap.check_time_and_version - Analyzing file at: [full_path]"
    # Or your script's "run_carball: Executing carball for [full_path]"
    
    # Let's assume your main processing script logs the full path when it starts processing a replay.
    # Example log message format that would be ideal:
    # "INFO - module:lineno - Processing C:\path\to\replays\subdir\failed_replay.replay..."
    
    # Search for lines indicating the start of processing for this specific replay name.
    # The key is that `replay_name` (e.g., "my_replay.replay") must be part of the logged path.
    path_pattern = re.compile(rf"Processing .*{re.escape(replay_name)}", re.IGNORECASE)
    # A more specific pattern if you know carball logs it:
    # carball_path_pattern = re.compile(rf"Analyzing file at: (.*{re.escape(replay_name)})", re.IGNORECASE)


    for line in reversed(log_lines): # Search backwards, assuming more recent relevant logs
        match = path_pattern.search(line)
        if match:
            # Try to extract the full path part
            # This depends heavily on your exact log message format for "Processing X"
            # Example: "INFO - module:lineno - Processing C:\path\to\file.replay..."
            path_extract_match = re.search(r"Processing (.*?" + re.escape(replay_name) + r")", line, re.IGNORECASE)
            if path_extract_match:
                found_path = path_extract_match.group(1).strip().split("...")[0] # Get part before "..." if present
                # Verify it looks like a real path
                if Path(found_path).name.lower() == replay_name.lower():
                    logging.debug(f"Path for {replay_name} found in log: {found_path}")
                    return found_path

    # Fallback: Search the parent_search_dir structure
    logging.debug(f"Path for {replay_name} not found in log via 'Processing ...' pattern. Searching filesystem under {parent_search_dir}")
    try:
        found_files = list(parent_search_dir.rglob(replay_name))
        if found_files:
            logging.debug(f"Path for {replay_name} found via rglob: {str(found_files[0])}")
            return str(found_files[0])
    except Exception as e:
        logging.warning(f"Error during rglob search for {replay_name} under {parent_search_dir}: {e}")
        
    logging.warning(f"Original path for {replay_name} could not be determined from logs or filesystem search.")
    return "Path not determined"


def extract_errors_for_replay(replay_name: str, log_lines: list[str]) -> list[str]:
    """
    Extracts relevant ERROR, CRITICAL, and WARNING messages for a given replay name from log lines.
    Also looks for common skip/empty INFO messages.
    """
    errors = []
    # Keywords indicating a reason for failure or being empty.
    # These should match prefixes or key phrases in your log messages.
    # The order can matter for how you want to prioritize.
    
    # Specific error patterns from your original script's logging:
    # "CRITICAL FAIL [replay_name]"
    # "carball.exe failed for [replay_name]"
    # "Unexpected error running carball for [replay_name]"
    # "metadata.json not found in [output_dir] for [replay_name]"
    # "__game.parquet not found for [replay_name]"
    # "__ball.parquet not found for [replay_name]"
    # "Error concatenating DataFrames for [replay_name]"
    # "Skipping replay [replay_name] due to player data processing issues" (often preceded by "Not 3v3")
    # "No valid frames remaining after null/initial processing for [replay_name]"
    # "All data removed after post-goal cleaning for [replay_name]"
    # "No data remaining after filtering by gameplay_periods for [replay_name]"
    # "No data remaining after downsampling for [replay_name]"
    # "[replay_name] processing returned None." (from main loop)
    # "[replay_name] processing resulted in an empty DataFrame." (from main loop)
    # "Exception during processing of [replay_name]" (from main loop)
    # "Exception for [replay_name] in main processing loop" (from main loop)

    # Search for lines containing the replay name AND an error/warning level or specific skip message
    relevant_lines_for_replay = [line for line in log_lines if replay_name in line]

    if not relevant_lines_for_replay:
        errors.append("No log entries found containing this replay name.")
        return errors

    primary_error_found = False
    for line_content in relevant_lines_for_replay:
        # Check for critical/error level messages first
        if "CRITICAL" in line_content or "ERROR" in line_content:
            match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - (?:CRITICAL|ERROR) - .*?:\d+ - (.*)', line_content)
            error_message = match.group(1).strip() if match else line_content.strip()
            errors.append(f"(CRITICAL/ERROR) {error_message}")
            primary_error_found = True # Prioritize this
            # If it's a "CRITICAL FAIL processing" message, that's usually the main one.
            if "CRITICAL failure processing" in error_message or "CRITICAL FAIL" in error_message:
                break # Stop collecting further errors for this replay if a critical processing failure is found

    if not primary_error_found: # If no critical/error, look for warnings or specific info skips
        for line_content in relevant_lines_for_replay:
            if "WARNING" in line_content:
                match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - WARNING - .*?:\d+ - (.*)', line_content)
                warn_message = match.group(1).strip() if match else line_content.strip()
                errors.append(f"(WARNING) {warn_message}")
            elif "INFO" in line_content: # Check for specific INFO messages indicating a skip/empty result
                skip_patterns = [
                    "Skipping replay.*player data issue",
                    "Not 3v3",
                    "No valid frames remaining",
                    "All data removed after post-goal cleaning",
                    "No data remaining after filtering by gameplay_periods",
                    "No data remaining after downsampling",
                    "processing returned None",
                    "resulted in an empty DataFrame"
                ]
                if any(pattern in line_content for pattern in skip_patterns):
                    match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - INFO - .*?:\d+ - (.*)', line_content)
                    info_message = match.group(1).strip() if match else line_content.strip()
                    errors.append(f"(INFO Skip/Empty) {info_message}")
    
    if not errors:
        errors.append("No specific error/warning/skip messages found in log entries for this replay, but it was listed as failed/empty.")
        
    return list(set(errors)) # Return unique error messages


def generate_failure_analysis_report(
    target_parent_dir: Path,
    summary_filename: str = "processing_summary.txt", # Default name
    log_filename: str = "replay_processor.log",       # Default name
    analysis_output_filename: str = "failure_analysis.txt"
):
    """
    Main function to generate the failure analysis report.
    """
    configure_analysis_logging()
    logging.info(f"Starting failure analysis for directory: {target_parent_dir}")

    summary_file_path = target_parent_dir / summary_filename
    log_file_path = target_parent_dir / log_filename # Assuming log is in parent_dir too
    analysis_output_path = target_parent_dir / analysis_output_filename

    if not summary_file_path.exists():
        logging.error(f"Processing summary file not found: {summary_file_path}")
        print(f"ERROR: Summary file not found at {summary_file_path}")
        return
    if not log_file_path.exists():
        logging.error(f"Replay processor log file not found: {log_file_path}")
        print(f"ERROR: Log file not found at {log_file_path}")
        return

    failed_replay_names = []
    try:
        with open(summary_file_path, 'r', encoding='utf-8') as f:
            in_failed_section = False
            for line in f:
                line_stripped = line.strip()
                # Accommodate different possible section headers
                if line_stripped.lower().startswith("--- failed/empty replays"):
                    in_failed_section = True
                    continue
                if in_failed_section:
                    if line_stripped.startswith("---"): # End of section
                        break
                    if line_stripped.startswith("- "):
                        failed_replay_names.append(line_stripped[2:].strip()) # Get name after "- "
    except Exception as e:
        logging.error(f"Error reading summary file {summary_file_path}: {e}")
        return

    if not failed_replay_names:
        logging.info("No failed replays listed in the summary file.")
        report_content = [
            "Failure Analysis Report",
            f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Target Directory: {target_parent_dir.resolve()}",
            f"Summary File: {summary_file_path.name}",
            f"Log File: {log_file_path.name}",
            "\nNo failed replays found in the summary file."
        ]
        with open(analysis_output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_content))
        logging.info(f"Empty failure analysis report written to: {analysis_output_path}")
        return
    
    logging.info(f"Found {len(failed_replay_names)} failed replay names in summary. Analyzing log entries...")

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_lines = f.readlines()
    except Exception as e:
        logging.error(f"Error reading log file {log_file_path}: {e}")
        return

    analysis_report_lines = [
        "========================================",
        " Failure Analysis Report",
        "========================================",
        f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Target Directory: {target_parent_dir.resolve()}",
        f"Summary File Used: {summary_file_path.name}",
        f"Log File Used: {log_file_path.name}",
        f"\nFound {len(failed_replay_names)} failed/empty replays listed in summary.",
        "----------------------------------------\n"
    ]

    for replay_name in failed_replay_names:
        analysis_report_lines.append(f"--- Replay File: {replay_name} ---")
        
        original_path = find_replay_original_path_in_log(replay_name, log_lines, target_parent_dir)
        analysis_report_lines.append(f"  Determined Original Path: {original_path}")
        
        errors = extract_errors_for_replay(replay_name, log_lines)
        if errors:
            analysis_report_lines.append(f"  Potential Reasons/Errors from Log ({len(errors)} unique entries):")
            for i, error_msg in enumerate(errors):
                analysis_report_lines.append(f"    {i+1}. {error_msg}")
        else:
            analysis_report_lines.append("  No specific error messages found in the log for this replay name.")
        analysis_report_lines.append("-" * 40 + "\n")

    try:
        with open(analysis_output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(analysis_report_lines))
        logging.info(f"Failure analysis report written to: {analysis_output_path}")
        print(f"SUCCESS: Failure analysis report written to: {analysis_output_path}")
    except IOError as e:
        logging.error(f"Failed to write failure analysis report to {analysis_output_path}: {e}")
        print(f"ERROR: Failed to write failure analysis report: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyzes processing logs to detail failed replay conversions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "parent_dir",
        type=Path,
        help="The main directory where replay processing was performed. "
             "This directory should contain the summary file and the main log file."
    )
    parser.add_argument(
        "--summary_file",
        type=str,
        default="processing_summary.txt", # Default name used by your main script
        help="Name of the processing summary text file."
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="replay_processor.log", # Default name used by your main script
        help="Name of the main replay_processor log file."
    )
    parser.add_argument(
        "--analysis_output",
        type=str,
        default="failure_analysis.txt",
        help="Name for the generated failure analysis report file."
    )
    args = parser.parse_args()

    # Ensure parent_dir exists
    if not args.parent_dir.is_dir():
        print(f"ERROR: Provided parent directory does not exist or is not a directory: {args.parent_dir}")
        sys.exit(1)

    generate_failure_analysis_report(
        args.parent_dir,
        args.summary_file,
        args.log_file,
        args.analysis_output
    )