import os
import pandas as pd
import matplotlib.pyplot as plt

# Specify the base directory where the 'output' folder is located
base_directory = 'G:/Projects/PycharmProjects/condaProject/output'

# Iterate through each subdirectory in the 'output' folder
for subdir, dirs, files in os.walk(base_directory):
    folder_name = os.path.basename(subdir)  # Extract the folder name
    for file in files:
        # Check if the file is a CSV file
        if file.endswith('.csv'):
            # Construct the full path to the file
            filepath = os.path.join(subdir, file)
            # Load the CSV data
            data = pd.read_csv(filepath)

            # Create the combined graph with both metrics
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))
            ax1.plot(data['Epoch'], data['Train Loss'], label='Train Loss', marker='o', color='blue')
            ax1.plot(data['Epoch'], data['Valid Loss'], label='Valid Loss', marker='o', color='red')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title(f'Training and Validation Loss - {folder_name}')
            ax1.legend(loc='upper right')
            ax1.grid(True)

            ax2.plot(data['Epoch'], data['Train Accuracy'], label='Train Accuracy', marker='s', linestyle='--',
                     color='blue')
            ax2.plot(data['Epoch'], data['Valid Accuracy'], label='Valid Accuracy', marker='s', linestyle='--',
                     color='red')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.set_title(f'Training and Validation Accuracy - {folder_name}')
            ax2.legend(loc='lower right')
            ax2.grid(True)
            combined_filename = os.path.join(subdir, 'redrawn_graph.png')
            plt.tight_layout()
            plt.savefig(combined_filename)
            plt.close()

            # Create and save the Loss plot separately
            plt.figure(figsize=(10, 6))
            plt.plot(data['Epoch'], data['Train Loss'], label='Train Loss', marker='o', color='blue')
            plt.plot(data['Epoch'], data['Valid Loss'], label='Valid Loss', marker='o', color='red')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training and Validation Loss - {folder_name}')
            plt.legend(loc='upper right')
            plt.grid(True)
            loss_filename = os.path.join(subdir, 'redrawn_loss_graph.png')
            plt.savefig(loss_filename)
            plt.close()

            # Create and save the Accuracy plot separately
            plt.figure(figsize=(10, 6))
            plt.plot(data['Epoch'], data['Train Accuracy'], label='Train Accuracy', marker='s', linestyle='--',
                     color='blue')
            plt.plot(data['Epoch'], data['Valid Accuracy'], label='Valid Accuracy', marker='s', linestyle='--',
                     color='red')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.title(f'Training and Validation Accuracy - {folder_name}')
            plt.legend(loc='lower right')
            plt.grid(True)
            accuracy_filename = os.path.join(subdir, 'redrawn_accuracy_graph.png')
            plt.savefig(accuracy_filename)
            plt.close()
