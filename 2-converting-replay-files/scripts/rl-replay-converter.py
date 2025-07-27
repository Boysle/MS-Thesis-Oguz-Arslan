import carball
import json
import gzip
import os
from carball.json_parser.game import Game
from carball.analysis.analysis_manager import AnalysisManager
from carball.analysis.utils.pandas_manager import PandasManager
from carball.analysis.utils.proto_manager import ProtobufManager

# Define the path to your replay file and the output JSON file
replay_file_path = r"C:\Users\Arslan\Desktop\c2076a12-ab3b-4a27-9469-dc8653070fc5.replay"
output_json_path = r"C:\Users\Arslan\Desktop\c2076a12-ab3b-4a27-9469-dc8653070fc5.json"

# Define output file paths for the analysis results
output_pts_path = r"C:\Users\Arslan\Desktop\output.pts"
output_gzip_path = r"C:\Users\Arslan\Desktop\output.gzip"

try:
    # Analyze the replay file
    analysis_manager = carball.analyze_replay_file(replay_file_path)

    # Get the JSON data from the analysis
    json_data = analysis_manager.get_json_data()

    # Save the JSON data to a file
    with open(output_json_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

    print(f"Analysis complete! JSON data saved to {output_json_path}")

    # DEBUGGING: Add checks to see if the data is correctly loaded
    print("Initializing game object with JSON data...")

    try:
        game = Game()
        game.initialize(loaded_json=json_data)
        print("Game object initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize game: {e}")
        raise

    try:
        print("Creating analysis manager...")
        analysis_manager = AnalysisManager(game)
        analysis_manager.create_analysis()
        print("Analysis created successfully!")
    except Exception as e:
        print(f"Error during analysis creation: {e}")
        raise

    # Return the proto object in Python
    try:
        print("Getting protobuf data...")
        proto_object = analysis_manager.get_protobuf_data()
        print("Protobuf data retrieved!")
    except Exception as e:
        print(f"Error retrieving protobuf data: {e}")
        raise

    # Return the pandas DataFrame in Python
    try:
        print("Getting pandas DataFrame...")
        dataframe = analysis_manager.get_data_frame()
        print("Pandas DataFrame retrieved!")
    except Exception as e:
        print(f"Error retrieving pandas DataFrame: {e}")
        raise

    # Continue with saving proto and pandas data as before...

    print("All analysis steps completed successfully!")

except FileNotFoundError as e:
    print(f"Error: {e}")
except TypeError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
