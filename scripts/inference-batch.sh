#!/bin/bash

# filepath: /Users/eric/Documents/mcomp-nus/CS5260/project/SMPLest-X/scripts/inference-batch.sh

# Check if a folder name is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <FOLDER_INSIDE_DEMO>"
  exit 1
fi

FOLDER_NAME="$1"
# Construct the full path assuming the folder is inside 'demo'
FOLDER="demo/$FOLDER_NAME"

# Check if the constructed path is a directory
if [ ! -d "$FOLDER" ]; then
  echo "Error: Directory '$FOLDER' not found."
  echo "Make sure '$FOLDER_NAME' exists inside the 'demo' directory."
  exit 1
fi

# Find all mp4 files in the specified folder and process them
find "$FOLDER" -maxdepth 1 -name "*.mp4" -print0 | while IFS= read -r -d $'\0' file; do
  echo "Processing $file..."
  # Assuming inference.sh is in the scripts directory relative to the project root
  # and can handle paths relative to the project root
  # Remove the 'demo/' prefix from the file path before passing it
  relative_file_path="${file#demo/}"
  echo "Running inference on $relative_file_path"
  sh scripts/inference.sh smplest_x_h "$relative_file_path" 30
done

echo "Batch processing complete for folder '$FOLDER'."