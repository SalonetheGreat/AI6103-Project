import torch
import torch.optim as optim
import torch.nn as nn
from data import get_train_valid_loader, get_test_loader
from mobilenet import MobileNet
import matplotlib.pyplot as plt
import os
import time
import logging
import csv
from datetime import timedelta

# Suppress torchvision logs (especially for downloading datasets)
logging.getLogger("torchvision").setLevel(logging.ERROR)


# Function to calculate the time taken per epoch or validation
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time / 60)
    seconds = int(elapsed_time % 60)
    return minutes, seconds


# Function to save training metrics to a CSV file
def save_training_metrics(output_dir, train_losses, valid_losses, train_accuracies, valid_accuracies, learning_rates):
    metrics_file = os.path.join(output_dir, 'training_metrics.csv')
    with open(metrics_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Valid Loss', 'Train Accuracy', 'Valid Accuracy', 'Learning Rate'])
        for epoch, (train_loss, valid_loss, train_acc, valid_acc, lr) in enumerate(
                zip(train_losses, valid_losses, train_accuracies, valid_accuracies, learning_rates), 1):
            writer.writerow([epoch, train_loss, valid_loss, train_acc, valid_acc, lr])
    print(f"Training metrics saved to {metrics_file}")


# Function to save the final model checkpoint
def save_final_model(output_dir, epoch, model, optimizer, train_loss, valid_loss):
    checkpoint_path = os.path.join(output_dir, f'mobilenet_final_epoch_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'valid_loss': valid_loss,
    }, checkpoint_path)
    print(f'Final model checkpoint saved: {checkpoint_path}')


# Training and validation process
def train_model_with_params(learning_rate, weight_decay, activation_function, scheduler_name, epochs, device,
                            train_loader, valid_loader):
    # Dynamic output directory based on hyperparameters
    output_dir = f'./output/lr_{learning_rate}_wd_{weight_decay}_{activation_function}_{scheduler_name}/'
    os.makedirs(output_dir, exist_ok=True)

    # Modify the model based on the activation function
    model = MobileNet(num_classes=100,
                      sigmoid_block_ind=[4, 5, 6, 7, 8, 9, 10] if activation_function == "sigmoid" else []).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

    # Learning rate scheduler (constant or cosine annealing)
    if scheduler_name == "cosine_annealing":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "constant":
        scheduler = None  # No scheduling, constant learning rate
    else:
        raise ValueError("Unsupported scheduler name. Use 'constant' or 'cosine_annealing'.")

    # Lists to track loss, accuracy, and learning rates
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []
    learning_rates = []

    overall_start_time = time.time()  # Track the overall start time

    def train(epoch):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        start_time = time.time()  # Start the timer for training

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to GPU
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100. * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        learning_rates.append(optimizer.param_groups[0]['lr'])  # Track the learning rate

        end_time = time.time()  # End the timer for training
        minutes, seconds = epoch_time(start_time, end_time)

        print(
            f'Epoch [{epoch}/{epochs}] | LR: {optimizer.param_groups[0]["lr"]} | Train Loss: {train_loss:.3f} | Train Accuracy: {train_accuracy:.3f}% | Training Time: {minutes}m {seconds}s')

    def validate(epoch):
        model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()  # Start the timer for validation

        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                valid_loss += loss.item()

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        valid_loss /= len(valid_loader)
        valid_accuracy = 100. * correct / total
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)

        end_time = time.time()  # End the timer for validation
        minutes, seconds = epoch_time(start_time, end_time)
        print(
            f'Epoch [{epoch}/{epochs}] | Valid Loss: {valid_loss:.3f} | Valid Accuracy: {valid_accuracy:.3f}% | Validation Time: {minutes}m {seconds}s')

    # Begin training
    for epoch in range(1, epochs + 1):
        train(epoch)
        validate(epoch)

        # Print the overall elapsed time after each epoch
        overall_end_time = time.time()
        overall_elapsed = timedelta(seconds=int(overall_end_time - overall_start_time))
        print(f"Total time elapsed after Epoch {epoch}: {overall_elapsed}")

        if scheduler:
            scheduler.step()  # Adjust the learning rate according to the scheduler

    # Save final training results
    save_training_metrics(output_dir, train_losses, valid_losses, train_accuracies, valid_accuracies, learning_rates)
    save_final_model(output_dir, epochs, model, optimizer, train_losses[-1], valid_losses[-1])

    # Plot loss, accuracy, and learning rate curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(valid_accuracies, label='Valid Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(learning_rates, label='Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()

    plt.savefig(os.path.join(output_dir, 'training_curves.png'))  # Save the plot to the output directory
    print(f'Training curves saved to {output_dir}')
    plt.show()


if __name__ == "__main__":
    # Hyperparameter settings to test
    learning_rates = [0.2, 0.05, 0.01]
    weight_decays = [5e-4, 1e-4]
    activation_functions = ['relu', 'sigmoid']
    scheduler_names = ['constant', 'cosine_annealing']
    epochs = 300

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load datasets
    data_dir = './data'
    batch_size = 128
    random_seed = 42
    valid_size = 0.2
    num_workers = 4  # Number of threads for data loading
    pin_memory = True  # Pin memory for faster GPU transfer
    train_loader, valid_loader = get_train_valid_loader(
        data_dir, batch_size, augment=True, random_seed=random_seed,
        valid_size=valid_size, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = get_test_loader(data_dir, batch_size, num_workers=num_workers, pin_memory=pin_memory)

    # Fixed values when testing other parameters
    fixed_learning_rate = 0.05
    fixed_activation_function = 'relu'
    fixed_scheduler_name = 'constant'

    # Test 1: Different Learning Rates (with no weight decay)
    print("\nTesting different learning rates (no weight decay)")
    for lr in learning_rates:
        train_model_with_params(lr, 0, fixed_activation_function, fixed_scheduler_name, epochs, device, train_loader,
                                valid_loader)

    # Test 2: Different Learning Rate Schedules (with no weight decay)
    print("\nTesting different learning rate schedules (no weight decay)")
    for scheduler_name in scheduler_names:
        train_model_with_params(fixed_learning_rate, 0, fixed_activation_function, scheduler_name, epochs, device,
                                train_loader, valid_loader)

    # Test 3: Different Weight Decays
    print("\nTesting different weight decay values")
    for wd in weight_decays:
        train_model_with_params(fixed_learning_rate, wd, fixed_activation_function, fixed_scheduler_name, epochs,
                                device, train_loader, valid_loader)

    # Test 4: Different Activation Functions (with no weight decay)
    print("\nTesting different activation functions (no weight decay)")
    for act_func in activation_functions:
        train_model_with_params(fixed_learning_rate, 0, act_func, fixed_scheduler_name, epochs, device, train_loader,
                                valid_loader)
