import os
import json
import numpy as np

# Define the directories
MUSIC_NPY_DIR = "music_data/music_npy"
LABEL_JSON_DIR = "music_data/label_json" # Replace with your desired output label directory path

# Create the output directory if it doesn't exist
os.makedirs(LABEL_JSON_DIR, exist_ok=True)

# Iterate through files in the npy directory
for filename in os.listdir(MUSIC_NPY_DIR):
    if filename.lower().endswith(".npy"):
        npy_path = os.path.join(MUSIC_NPY_DIR, filename)
        # Construct the output JSON filename
        base_filename = os.path.splitext(filename)[0]
        json_filename = f"{base_filename}.json"
        json_path = os.path.join(LABEL_JSON_DIR, json_filename)

        print(f"Processing {filename} -> {json_filename}...")

        try:
            # Load the numpy array to get its shape (assuming frames is the first dimension)
            music_data = np.load(npy_path)
            num_frames = music_data.shape[0] # Or adjust based on how 'frames' is defined in your data

            # Create the label dictionary
            label_data = {
                "name": base_filename,
                "style1": "Solo",
                "style2": "Kpop",
                "frames": num_frames
            }

            # Write the dictionary to a JSON file
            with open(json_path, 'w') as json_file:
                json.dump(label_data, json_file, indent=4)

            print(f"Successfully created {json_filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

print("Label generation process finished.")