import requests
import os
import time
import statistics
from datetime import datetime

def get_user_input():
    """Get group ID and name from user with validation and confirmation."""
    print("\n" + "="*50)
    print("Rocket League Replay Downloader".center(50))
    print("="*50 + "\n")
    
    # Get group ID
    while True:
        group_id = input("Enter the group ID (or press Enter to quit): ").strip()
        if not group_id:
            print("Exiting...")
            exit()
        if len(group_id) < 5:  # Basic validation
            print("Error: Group ID seems too short. Please try again.")
            continue
        break
    
    # Get group name
    group_name = input("Enter a name for this group (for folder/log naming): ").strip()
    if not group_name:
        group_name = "Unnamed_Group"
    
    # Get download directory
    default_dir = r"D:\\Raw RL Esports Replays\\Extra Replays\\RLCS 2024\\World Championship (Fort Worth)"
    download_dir = input(f"Enter download directory (default: {default_dir}): ").strip()
    if not download_dir:
        download_dir = default_dir
    
    # Confirmation
    print("\n" + "-"*50)
    print(f"Group ID: {group_id}")
    print(f"Group Name: {group_name}")
    print(f"Download Directory: {download_dir}")
    print("-"*50 + "\n")
    
    confirm = input("Proceed with these settings? (y/n): ").lower()
    if confirm != 'y':
        print("Aborted by user.")
        exit()
    
    return group_id, group_name, download_dir

# Get user input at start
group_id, group_name, desired_directory = get_user_input()

# Example forms of group id, group name, and desired directory
# group_id = 'day-3-swiss-stage-4l511c4nfg'
# group_name = 'Day 3 Swiss Stage'
# desired_directory = 'E:\\Raw RL Esports Replays'

# API key varies for each user and can be generated in https://ballchasing.com/upload
api_key = 'PfRoIALT3dfYslK0n0sKsGhar3Dh9zqVUaW7gwyW'

# Rate limiting configuration
RATE_LIMIT_RETRIES = 5  # Number of times to retry when hitting rate limits
RATE_LIMIT_DELAY = 1    # Initial delay in seconds (will increase with each retry)
MAX_REQUESTS_PER_SECOND = 2  # API limit of 2 calls/second

# Define the API endpoint URL
replay_base_url = 'https://ballchasing.com/api/replays'
group_base_url = 'https://ballchasing.com/api/groups?group'
leaf_group_base_url = 'https://ballchasing.com/api/replays?group'

# Define headers with the Authorization header containing the API key
headers = {
    'Authorization': f'{api_key}'
}

# Initialize statistics variables
total_replays = 0
file_sizes = []
file_info = []  # Store tuples of (replay_id, size, path)
start_time = None
process_complete = False
last_request_time = 0

def enforce_rate_limit():
    """Ensure we don't exceed the API rate limit of 2 calls/second."""
    global last_request_time
    elapsed = time.time() - last_request_time
    if elapsed < 1/MAX_REQUESTS_PER_SECOND:
        time.sleep((1/MAX_REQUESTS_PER_SECOND) - elapsed)
    last_request_time = time.time()

def make_api_request(url, max_retries=RATE_LIMIT_RETRIES):
    """Make an API request with rate limit handling and retries."""
    for attempt in range(max_retries):
        enforce_rate_limit()
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 429:
                wait_time = RATE_LIMIT_DELAY * (attempt + 1)
                print(f"Rate limit hit (attempt {attempt + 1}/{max_retries}). Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            return response
        except requests.exceptions.RequestException as e:
            print(f"Request failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
            time.sleep(RATE_LIMIT_DELAY * (attempt + 1))
    return None

def write_summary_log(root_path, group_name, total_time, total_replays, file_info):
    """Write a detailed summary log file with all the collected statistics."""
    log_filename = f"{group_name}_download_summary.log"
    log_path = os.path.join(root_path, log_filename)
    
    # Calculate statistics
    sizes = [info[1] for info in file_info if info[1] is not None]
    total_size = sum(sizes) / (1024 * 1024) if sizes else 0  # Convert to MB
    avg_size = statistics.mean(sizes) / (1024 * 1024) if sizes else 0  # Convert to MB
    size_sd = statistics.stdev(sizes) / (1024 * 1024) if len(sizes) > 1 else 0  # Convert to MB
    
    # Convert total_time to HH:MM:SS format
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    
    # Find min and max size replays
    min_size_info = min(file_info, key=lambda x: x[1] if x[1] is not None else float('inf'))
    max_size_info = max(file_info, key=lambda x: x[1] if x[1] is not None else float('-inf'))
    
    with open(log_path, 'w') as log_file:
        log_file.write(f"Rocket League Replay Download Summary - {group_name}\n")
        log_file.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Total processing time: {total_time:.2f} seconds\n")
        log_file.write(f"Time taken: {time_str} (HH:MM:SS)\n")  # New line added here
        log_file.write(f"Total replays attempted: {total_replays}\n")
        log_file.write(f"Total replays downloaded: {len(sizes)}\n")
        log_file.write(f"Total size of all replays: {total_size:.2f} MB\n")
        log_file.write(f"Average replay size: {avg_size:.2f} MB\n")
        log_file.write(f"Standard deviation of replay sizes: {size_sd:.2f} MB\n")
        
        if sizes:
            log_file.write("\n--- Size Extremes ---\n")
            log_file.write(f"Smallest replay: {min_size_info[0]}.replay ({min_size_info[1]/(1024*1024):.2f} MB)\n")
            log_file.write(f"Path: {os.path.join(min_size_info[2], min_size_info[0] + '.replay')}\n")
            log_file.write(f"\nLargest replay: {max_size_info[0]}.replay ({max_size_info[1]/(1024*1024):.2f} MB)\n")
            log_file.write(f"Path: {os.path.join(max_size_info[2], max_size_info[0] + '.replay')}\n")
        
        # List failed downloads if any
        failed_downloads = [info for info in file_info if info[1] is None]
        if failed_downloads:
            log_file.write(f"\n--- Failed Downloads ({len(failed_downloads)}) ---\n")
            for failed in failed_downloads:
                log_file.write(f"- {failed[0]}.replay (Path: {os.path.join(failed[2], failed[0] + '.replay')})\n")
        else:
            log_file.write(f"\n--- All Downloads Successful ---\n")
    
    print(f"Summary log written to: {log_path}")

def download_replay_file(replay_id, path):
    global total_replays, file_info
    
    # Create full file path
    file_path = os.path.join(path, f'{replay_id}.replay')
    
    # Check if file already exists
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        print(f'Replay {replay_id}.replay already exists ({file_size/(1024*1024):.2f} MB) - skipping')
        file_info.append((replay_id, file_size, path))  # Track existing files too
        return True  # Consider this a "success" since we have the file
    
    url = f'{replay_base_url}/{replay_id}/file'
    response = make_api_request(url)
    
    if response and response.status_code == 200:
        file_size = len(response.content)
        
        # Double-check directory exists before writing
        os.makedirs(path, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            f.write(response.content)
            print(f'Downloaded {replay_id}.replay ({file_size/(1024*1024):.2f} MB)')
        
        file_info.append((replay_id, file_size, path))
        total_replays += 1
        return True
    else:
        print(f'Failed to download {replay_id}.replay')
        file_info.append((replay_id, None, path))  # Track failure
        total_replays += 1
        return False

def access_replay_from_leaf_group(group_id, path):
    print(f'Accessing replays from group: {group_id}')
    url = f'{leaf_group_base_url}={group_id}'
    
    response = make_api_request(url)
    if response and response.status_code == 200:
        replay_ids = [replay['id'] for replay in response.json()['list']]
        for replay_id in replay_ids:
            if replay_id:
                print(f'Processing replay: {replay_id}')
                download_replay_file(replay_id, path)
            else:
                print('No ID found for this replay')
    else:
        print(f'Failed to access leaf group {group_id}')

def sanitize_folder_name(name):
    """Remove characters that are invalid in Windows filenames."""
    # List of invalid characters in Windows filenames
    invalid_chars = '<>:"/\\|?*'
    
    # Also remove control characters (0-31) and DEL (127)
    sanitized = ''.join(
        char for char in name 
        if char not in invalid_chars and ord(char) >= 32 and ord(char) != 127
    )
    
    # Remove leading/trailing spaces and dots (Windows doesn't like these)
    sanitized = sanitized.strip(' .')
    
    # Replace colons with a dash (common case we want to handle nicely)
    sanitized = sanitized.replace(':', ' -')
    
    # If we end up with an empty string, return a default
    if not sanitized:
        sanitized = "Unnamed Folder"
    
    return sanitized

def create_folder_tree_and_access_replays(group_id, parent_path, group_name):
    global start_time, process_complete
    
    # Start timer on first call
    if start_time is None:
        start_time = time.time()
        print("Download process started...")
    
    # Sanitize the group name before using it as a folder name
    safe_group_name = sanitize_folder_name(group_name)
    
    # Make a GET request to retrieve the group info
    child_group_info_url = f'{group_base_url}={group_id}'
    response = make_api_request(child_group_info_url)
    
    if response and response.status_code == 200:
        group_info = response.json()
        
        try:
            # Create the directory with sanitized name
            current_path = os.path.join(parent_path, safe_group_name)
            os.makedirs(current_path, exist_ok=True)
            
            # Process child groups if they exist
            child_groups = group_info.get('list', [])
            if child_groups:
                # Recursively process child groups
                for child_group in child_groups:
                    child_group_id = child_group.get('id')
                    original_child_name = child_group.get('name')
                    if child_group_id and original_child_name:
                        safe_child_name = sanitize_folder_name(original_child_name)
                        create_folder_tree_and_access_replays(
                            child_group_id, 
                            current_path, 
                            safe_child_name
                        )
            else:
                # Leaf group reached
                access_replay_from_leaf_group(group_id, current_path)
                
        except (OSError, FileNotFoundError) as e:
            print(f"Warning: Could not create directory '{current_path}' - {str(e)}")
            print("Skipping this branch of the folder tree")
            return
            
        # Write summary when processing complete
        if not process_complete and parent_path == desired_directory:
            process_complete = True
            total_time = time.time() - start_time
            write_summary_log(parent_path, group_name, total_time, total_replays, file_info)
    else:
        print(f'Failed to retrieve group info for {group_id}')

# Create root directory if it doesn't exist
os.makedirs(desired_directory, exist_ok=True)

# Start the download process
create_folder_tree_and_access_replays(group_id, desired_directory, group_name)