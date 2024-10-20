import os
import shutil

# Define the relative paths from the script's location
source_directory = 'output'
destination_directory = 'graphs'

# Ensure the current directory is the project root (where the script is located)
project_root = os.path.dirname(os.path.abspath(__file__))  # Gets the directory where the script is located
source_path = os.path.join(project_root, source_directory)
destination_path = os.path.join(project_root, destination_directory)

# Create the destination directory if it doesn't exist
os.makedirs(destination_path, exist_ok=True)

# Iterate through each subdirectory in the source directory
for subdir, dirs, files in os.walk(source_path):
    for file in files:
        if file.endswith('.png'):
            # Construct the full path to the source file
            source_file_path = os.path.join(subdir, file)
            # Construct the full path to the destination file
            destination_file_path = os.path.join(destination_path, file)

            # Copy the file to the destination directory, overwriting any existing file
            shutil.copy2(source_file_path, destination_file_path)
            print(f"Copied {source_file_path} to {destination_file_path}")
