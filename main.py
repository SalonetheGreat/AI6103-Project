import os

# Specify the base directory where the 'output' folder is located
base_directory = 'G:/Projects/PycharmProjects/condaProject/output'

# Iterate through each subdirectory in the 'output' folder
for subdir, dirs, files in os.walk(base_directory):
    for file in files:
        # Check if the file name starts with 'redrawn_'
        if file.startswith('redrawn_'):
            # Construct the full path to the file
            file_path = os.path.join(subdir, file)
            # Remove the file
            os.remove(file_path)
            print(f"Removed {file_path}")
