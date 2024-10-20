import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt

# Specify the relative paths from the script's location
source_directory = 'output'
graphs_directory = 'graphs'

# Ensure the current directory is the project root (where the script is located)
project_root = os.path.dirname(os.path.abspath(__file__))  # Gets the directory where the script is located
source_path = os.path.join(project_root, source_directory)
graphs_path = os.path.join(project_root, graphs_directory)

# Create the graphs directory if it doesn't exist or clear it if it does
if not os.path.exists(graphs_path):
    os.makedirs(graphs_path, exist_ok=True)
else:
    # Remove all files in the graphs directory before generating new ones
    for filename in os.listdir(graphs_path):
        file_path = os.path.join(graphs_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove the file
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove the directory
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

# Iterate through each subdirectory in the 'output' folder
for subdir, dirs, files in os.walk(source_path):
    folder_name = os.path.basename(subdir)  # Extract the folder name
    for file in files:
        if file.endswith('.csv'):
            # Construct the full path to the file
            filepath = os.path.join(subdir, file)
            # Load the CSV data
            data = pd.read_csv(filepath)

            # Create and save the Loss graph
            plt.figure(figsize=(10, 6))
            plt.plot(data['Epoch'], data['Train Loss'], label='Train Loss', color='blue', marker='o')
            plt.plot(data['Epoch'], data['Valid Loss'], label='Valid Loss', color='red', marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'{folder_name} Training and Validation Loss')
            plt.legend(loc='upper right')
            plt.grid(True)
            loss_filename = os.path.join(graphs_path, f'{folder_name}_loss_graph.png')
            plt.savefig(loss_filename)
            plt.close()

            # Create and save the Accuracy graph
            plt.figure(figsize=(10, 6))
            plt.plot(data['Epoch'], data['Train Accuracy'], label='Train Accuracy', color='green', linestyle='--',
                     marker='s')
            plt.plot(data['Epoch'], data['Valid Accuracy'], label='Valid Accuracy', color='purple', linestyle='--',
                     marker='s')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.title(f'{folder_name} Training and Validation Accuracy')
            plt.legend(loc='lower right')
            plt.grid(True)
            accuracy_filename = os.path.join(graphs_path, f'{folder_name}_accuracy_graph.png')
            plt.savefig(accuracy_filename)
            plt.close()

            # Create and save the combined Loss and Accuracy graph
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(data['Epoch'], data['Train Loss'], label='Train Loss', color='blue', marker='o')
            ax1.plot(data['Epoch'], data['Valid Loss'], label='Valid Loss', color='red', marker='o')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax2 = ax1.twinx()
            ax2.plot(data['Epoch'], data['Train Accuracy'], label='Train Accuracy', color='green', linestyle='--',
                     marker='s')
            ax2.plot(data['Epoch'], data['Valid Accuracy'], label='Valid Accuracy', color='purple', linestyle='--',
                     marker='s')
            ax2.set_ylabel('Accuracy (%)', color='green')
            ax2.tick_params(axis='y', labelcolor='green')
            plt.title(f'{folder_name} Combined Loss and Accuracy')
            fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
            combined_filename = os.path.join(graphs_path, f'{folder_name}_combined_loss_accuracy_graph.png')
            plt.savefig(combined_filename)
            plt.close()
