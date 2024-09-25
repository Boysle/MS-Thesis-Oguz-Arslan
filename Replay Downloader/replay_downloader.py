import requests
import os

# API key can be generated in https://ballchasing.com/upload
api_key = 'PfRoIALT3dfYslK0n0sKsGhar3Dh9zqVUaW7gwyW'
group_id = 'swiss-wfsihlwi8v'
group_name = 'Swiss'
desired_directory = 'E:\RL Esports Replays'

# Define the API endpoint URL
replay_base_url = 'https://ballchasing.com/api/replays'
group_base_url = 'https://ballchasing.com/api/groups?group'
leaf_group_base_url = 'https://ballchasing.com/api/replays?group'


# Define headers with the Authorization header containing the API key
headers = {
    'Authorization': f'{api_key}'
}

# Function to download the replay file
def download_replay_file(replay_id, path):
    url = f'{replay_base_url}/{replay_id}/file'
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        # Write the binary contents to a file
        with open(f'{path}\{replay_id}.replay', 'wb') as f:
            f.write(response.content)
            print(f'Replay file {replay_id}.replay downloaded successfully.')
    else:
        print(f'Failed to download replay file. Status code: {response.status_code}')

def access_replay_from_leaf_group(group_id, path):
    print('Accessing replay: ', group_id)
    url = f'{leaf_group_base_url}={group_id}'
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        replay_ids = [replay['id'] for replay in response.json()['list']]
        for replay_id in replay_ids:
            if replay_id:
                print(replay_id)
                download_replay_file(replay_id, path)
            else:
                print('No ID found for this replay')
    else:
        print(f'Failed to find leaf group. Status code: {response.status_code}')


# Function to recursively create the folder tree and access replays from leaf groups
def create_folder_tree_and_access_replays(group_id, parent_path, group_name):
    # Make a GET request to retrieve the group info
    child_group_info_url = f'{group_base_url}={group_id}'
    response = requests.get(child_group_info_url, headers=headers)
    
    if response.status_code == 200:
        group_info = response.json()
        
        # Create the directory for the current group
        current_path = os.path.join(parent_path, group_name)
        os.makedirs(current_path, exist_ok=True)
        
        # Process child groups if they exist
        child_groups = group_info.get('list', [])
        if child_groups:
            # Recursively process child groups
            for child_group in child_groups:
                child_group_id = child_group.get('id')
                child_group_name = child_group.get('name')
                create_folder_tree_and_access_replays(child_group_id, current_path, child_group_name)
        else:
            # Leaf group reached, call access_replay_from_leaf_group function
            access_replay_from_leaf_group(group_id, current_path)
    else:
        print(f'Failed to retrieve group info. Status code: {response.status_code}')



# Example usage
create_folder_tree_and_access_replays(group_id, desired_directory, group_name)
