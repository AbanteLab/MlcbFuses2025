#!/usr/bin/env python3

### FUNCTIONS FOR GNN SCRIPT ###

import torch
import random
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.manifold import TSNE
import os
from torch_geometric.data import Data, Dataset
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix

# Dataset class
class AOGraphDataset(Dataset):
    def __init__(self, gene_features_all, sex_all, cag_all, y_all, edge_index, edge_attr):
        self.gene_features_all = gene_features_all
        self.sex_all = sex_all
        self.cag_all = cag_all
        self.y_all = y_all
        self.edge_index = edge_index
        self.edge_attr = edge_attr

    def __len__(self):
        return len(self.gene_features_all)

    def __getitem__(self, idx):
        
        x = self.gene_features_all[idx]  # Node features for sample idx
        y = self.y_all[idx]  # Target value for sample idx
        sex = self.sex_all[idx]
        cag = self.cag_all[idx]
        
        data = Data(
            x=x,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            y=y,
            sex=sex,
            cag=cag
        )
        return data

# Global variable for the log file
log_file_path = None

n_genes = None

def initialize_logger(output_dir):
    """Initializes the logger by creating a log file."""
    global log_file_path
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = os.path.join(output_dir, f'log_{timestamp}.txt')

    # Create the log file
    with open(log_file_path, 'w') as f:
        f.write(f"Log file created on {timestamp}\n")

def _print(*args, **kw):
    """Custom print function that logs to console and file."""
    message = "[%s] %s" % (datetime.now(), " ".join(map(str, args)))
    print(message, **kw)  # Print to console
    if log_file_path:  # Log to file if path is initialized
        with open(log_file_path, 'a') as log_file:
            log_file.write(message + '\n')
            
def set_global_ngenes(ngenes):
    global n_genes
    n_genes = ngenes    

def plot_learning_curves(train_losses, val_losses):
    """
    Generates a learning curves plot for training and validation losses.

    Parameters:
        train_losses (list or array-like): List of training loss values for each epoch.
        val_losses (list or array-like): List of validation loss values.

    Returns:
        fig (matplotlib.figure.Figure): Figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = len(train_losses)

    # Plot training loss
    ax.plot(range(1, epochs + 1), train_losses, label='Training Loss')

    # Plot validation loss every 5 epochs
    ax.plot(range(10, epochs + 1, 10), val_losses[9::10], label='Validation Loss', linestyle='--')

    # Set labels and title
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')

    # Add legend and grid
    ax.legend()
    ax.grid()

    # Return the figure object
    return fig
    
def plot_model_metrics(train_y, train_yp, test_y, test_yp, class_names=None):
    """
    Evaluates predicted labels against true labels and returns:
    - Accuracy
    - Classification report
    - Confusion matrix visualization
    
    Parameters:
        true_labels (torch.Tensor): True labels.
        predicted_labels (torch.Tensor): Predicted labels.
        class_names (list or None): List of class names for the classification report and confusion matrix. None if not provided.
    Returns:
        metrics_string (str): Text summary of accuracy and classification report.
        fig (matplotlib.figure.Figure): Figure object containing the confusion matrix plot.
    """
    # Convert tensors to numpy arrays
    train_y = train_y.numpy()
    train_yp = train_yp.numpy()
    test_y = test_y.numpy()
    test_yp = test_yp.numpy()

    # Compute metrics
    train_accuracy = balanced_accuracy_score(train_y, train_yp)
    train_conf_matrix = confusion_matrix(train_y, train_yp)
    test_accuracy = balanced_accuracy_score(test_y, test_yp)
    test_conf_matrix = confusion_matrix(test_y, test_yp)
    class_report = classification_report(test_y, test_yp, target_names=class_names)

    # Create metrics string
    metrics_string = f"Model Metrics:\n"
    metrics_string += f"Accuracy: {test_accuracy:.4f}\n"
    metrics_string += f"Train Balanced Accuracy: {train_accuracy:.4f}\n"
    metrics_string += f"Classification Report:\n{class_report}\n"
    metrics_string += f"Train confusion matrix:\n{train_conf_matrix}\n"
    metrics_string += f"Test confusion matrix:\n{test_conf_matrix}\n"

    # Create figure for confusion matrix
    test_conf_fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(test_conf_matrix, cmap=plt.cm.Blues)
    test_conf_fig.colorbar(cax)

    # Set axis labels
    if class_names:
        ax.set_xticklabels([''] + class_names, rotation=45)
        ax.set_yticklabels([''] + class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')

    # Annotate each cell with the numeric value
    for i in range(test_conf_matrix.shape[0]):
        for j in range(test_conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=test_conf_matrix[i, j], va='center', ha='center', color='black')

    # Apply tight layout
    test_conf_fig.tight_layout()

    # Create figure for training confusion matrix
    train_conf_fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(train_conf_matrix, cmap=plt.cm.Blues)
    train_conf_fig.colorbar(cax)

    # Set axis labels
    if class_names:
        ax.set_xticklabels([''] + class_names, rotation=45)
        ax.set_yticklabels([''] + class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')

    # Annotate each cell with the numeric value
    for i in range(train_conf_matrix.shape[0]):
        for j in range(train_conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=train_conf_matrix[i, j], va='center', ha='center', color='black')

    # Apply tight layout
    train_conf_fig.tight_layout()

    return metrics_string, test_conf_fig, train_conf_fig

def generate_intermediate_tsne_plot(model, test_loader, device, num_classes=5):
    """
    This function runs the forward pass on the test set, extracts intermediate features, 
    and creates a t-SNE plot colored by the predicted values.
    Needs output of a linear layer to be the first object saved in the model's 2nd output (list)

    Args:
    - model (torch.nn.Module): The trained model.
    - test_loader (torch.utils.data.DataLoader): The DataLoader for the test set.
    - device (torch.device): The device (CPU or CUDA) for model and data.
    - num_classes (int): The number of discrete classes for coloring.

    Returns:
    - matplotlib.figure.Figure: The generated t-SNE plot figure.
    """
    # Lists to store intermediate values and outputs
    all_x_lin1 = []
    all_outputs = []
    all_true_labels = []

    # Set the model to evaluation mode
    model.eval()

    # Iterate over the test_loader
    for data in test_loader:

        # Ensure the data is on the correct device (e.g., CUDA or CPU)
        data = data.to(device)

        # Perform the forward pass and capture the output
        with torch.no_grad():  # No need for gradients during inference
            output, extras_list = model(data)
        x_lin1 = extras_list[0]

        # Extract true labels from the batch
        true_labels = data.y

        # Append the results to the lists
        all_x_lin1.append(x_lin1.cpu().numpy())  # Move to CPU and convert to numpy array
        all_outputs.append(output.cpu().numpy())
        all_true_labels.append(true_labels.cpu().numpy())

    # Convert all lists to numpy arrays for easier handling
    all_x_lin1 = np.concatenate(all_x_lin1, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_true_labels = np.concatenate(all_true_labels, axis=0)

    # Perform t-SNE on the x_lin1 (the output of the first linear layer)
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(all_x_lin1)

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Define a discrete colormap
    cmap = plt.get_cmap('Accent', num_classes)

    # Plot 1: t-SNE colored by predicted labels
    predicted_labels = np.argmax(all_outputs, axis=1)
    scatter1 = axs[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=predicted_labels, cmap=cmap, alpha=0.8)
    axs[0].set_title('t-SNE plot (Predicted Labels)')
    axs[0].set_xlabel('t-SNE component 1')
    axs[0].set_ylabel('t-SNE component 2')

    # Plot 2: t-SNE colored by true labels
    scatter2 = axs[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=all_true_labels, cmap=cmap, alpha=0.8)
    axs[1].set_title('t-SNE plot (True Labels)')
    axs[1].set_xlabel('t-SNE component 1')
    axs[1].set_ylabel('t-SNE component 2')

    # Create a shared legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), markersize=10) for i in range(num_classes)]
    labels = [f'Class {i}' for i in range(num_classes)]
    fig.legend(handles, labels, loc='lower center', ncol=num_classes)

    # Adjust layout
    plt.tight_layout()

    # Return the figure containing both plots
    return fig
