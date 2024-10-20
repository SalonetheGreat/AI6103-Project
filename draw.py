import os
import pandas as pd
import matplotlib.pyplot as plt

# Specify the relative paths from the script's location
source_directory = 'output'

# Ensure the current directory is the project root (where the script is located)
project_root = os.path.dirname(os.path.abspath(__file__))  # Gets the directory where the script is located
source_path = os.path.join(project_root, source_directory)

# Iterate through each subdirectory in the 'output' folder
for subdir, dirs, files in os.walk(source_path):
    folder_name = os.path.basename(subdir)  # Extract the folder name
    for file in files:
        # Check if the file is a CSV file
        if file.endswith('.csv'):
            # Construct the full path to the file
            filepath = os.path.join(subdir, file)
            # Load the CSV data
            data = pd.read_csv(filepath)

            # Create the combined graph with both metrics in two separate plots
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))
            ax1.plot(data['Epoch'], data['Train Loss'], label='Train Loss', marker='o', color='blue')
            ax1.plot(data['Epoch'], data['Valid Loss'], label='Valid Loss', marker='o', color='red')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title(f'{folder_name} Training and Validation Loss')
            ax1.legend(loc='upper right')
            ax1.grid(True)

            ax2.plot(data['Epoch'], data['Train Accuracy'], label='Train Accuracy', marker='s', linestyle='--',
                     color='blue')
            ax2.plot(data['Epoch'], data['Valid Accuracy'], label='Valid Accuracy', marker='s', linestyle='--',
                     color='red')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.set_title(f'{folder_name} Training and Validation Accuracy')
            ax2.legend(loc='lower right')
            ax2.grid(True)
            combined_filename = os.path.join(subdir, f'{folder_name}_graph.png')
            plt.tight_layout()
            plt.savefig(combined_filename)
            plt.close()

            # Create the single combined graph for both loss and accuracy
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(data['Epoch'], data['Train Loss'], label='Train Loss', color='blue', marker='o')
            ax.plot(data['Epoch'], data['Valid Loss'], label='Valid Loss', color='red', marker='o')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss', color='blue')
            ax.tick_params(axis='y', labelcolor='blue')

            ax2 = ax.twinx()
            ax2.plot(data['Epoch'], data['Train Accuracy'], label='Train Accuracy', color='green', linestyle='--',
                     marker='s')
            ax2.plot(data['Epoch'], data['Valid Accuracy'], label='Valid Accuracy', color='purple', linestyle='--',
                     marker='s')
            ax2.set_ylabel('Accuracy (%)', color='green')
            ax2.tick_params(axis='y', labelcolor='green')

            plt.title(f'{folder_name} Combined Loss and Accuracy')
            fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
            combined_loss_accuracy_filename = os.path.join(subdir, f'{folder_name}_combined_loss_accuracy.png')
            plt.savefig(combined_loss_accuracy_filename)
            plt.close()
