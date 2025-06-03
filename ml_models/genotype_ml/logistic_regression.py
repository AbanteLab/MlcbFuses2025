#!/usr/bin/env python3

### Logistic regression ###
#%%

import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import sys
from itertools import product
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import balanced_accuracy_score
import argparse

PROJECT_ROOT = "/pool01/code/projects/abante_lab/ao_prediction_enrollhd_2024/ml_models"
# PROJECT_ROOT = "/gpfs/projects/ub212/ao_prediction_enrollhd_2024/code/src/ml_models/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load custom functions
from utils.evaluating_functions import save_results
from utils.data_loading import _print, load_X_y

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train a multinomial logistic regression model.")
parser.add_argument("--seednum", type=int, required=True, help="Seed number for data splitting")
parser.add_argument("--dataset", type=str, choices=['b1', 'b2'], required=True, help="Dataset to train on (must be 'b1' or 'b2')")
args = parser.parse_args()

_print("Start time")

#--------# Directories #--------#

# Change working directory
os.chdir('/pool01/projects/abante_lab/ao_prediction_enrollhd_2024/')
# os.chdir('/gpfs/projects/ub212/ao_prediction_enrollhd_2024/data/')

# Data directory
data_dir = "features/"

# Input files
if args.dataset == 'b1':
    X_path = data_dir + "scag_feature_matrix_m3_filt_0.01.npz"
elif args.dataset == 'b2':
    X_path = data_dir + "2d_gene_tensor_vae_snps.pth"
y_path = data_dir + "binned_ao.txt"

# Results directory
results_dir = "ml_results/classification/"

#--------# Load data #--------#

# Load X and y
X, y = load_X_y(X_path, y_path)

#%%
# Implementation with early stopping
class MultinomialLogisticRegression:
    def __init__(self, num_features, num_classes, lr=0.001, epochs=10000, lambda_l1=0.001, patience=10, device='cpu'):
        """
        Initialize the model with weights, biases, and hyperparameters.
        """
        self.lr = lr
        self.epochs = epochs
        self.num_classes = num_classes
        self.lambda_l1 = lambda_l1
        self.device = device
        self.patience = patience
        self.weights = torch.zeros((num_features, num_classes), device=self.device, dtype=torch.float32)
        torch.nn.init.xavier_uniform_(self.weights)
        self.biases = torch.zeros(num_classes, device=self.device, dtype=torch.float32)
        self.train_loss_history = []
        self.val_loss_history = []

    def reset_history(self):
        """
        Reset the training and validation loss history.
        """
        self.train_loss_history = []
        self.val_loss_history = []

    def get_params(self):
        """
        Get model parameters as a dictionary.
        """
        return {
            "num_features": self.weights.shape[0],
            "num_classes": self.num_classes,
            "lr": self.lr,
            "epochs": self.epochs,
            "lambda_l1": self.lambda_l1,
            "device": self.device
        }

    def _compute_logits(self, X):
        """Compute logits: W^T X + b."""
        logits = torch.matmul(X, self.weights) + self.biases
        return logits

    def _compute_probabilities(self, logits):
        """Apply softmax to logits to compute probabilities."""
        return torch.softmax(logits, dim=1)
    
    def _compute_loss(self, probabilities, y):
        """Compute loss (negative log-likelihood plus L1 regularization)."""
        log_probs = torch.log(probabilities + 1e-10)
        nll = - torch.sum(log_probs[torch.arange(y.shape[0]), y]) / y.shape[0]
        l1_reg = self.lambda_l1 * torch.sum(torch.abs(self.weights))
        return nll + l1_reg
    
    def _evaluate_loss(self, X, y):
        """Evaluate loss on validation set."""
        X, y = X.to(self.device), y.to(self.device)
        logits = self._compute_logits(X)
        probabilities = self._compute_probabilities(logits)
        loss = self._compute_loss(probabilities, y)
        return loss.item()

    def train(self, X, y, X_val=None, y_val=None):
        """
        Train the model with early stopping.
        """
        self.reset_history()
        X, y = X.to(self.device), y.to(self.device)

        patience_counter = 0
        best_val_loss = float('inf')

        for epoch in range(self.epochs):
            logits = self._compute_logits(X)
            probabilities = self._compute_probabilities(logits)
            one_hot_labels = torch.zeros((X.shape[0], self.num_classes), device=self.device)
            one_hot_labels[torch.arange(X.shape[0]), y] = 1
            
            # Compute gradients
            grad_weights = torch.matmul(X.T, probabilities - one_hot_labels) / X.shape[0]
            grad_biases = torch.mean(probabilities - one_hot_labels, dim=0)
            grad_l1 = self.lambda_l1 * torch.sign(self.weights)
            
            # update parameters
            self.weights -= self.lr * (grad_weights + grad_l1)
            self.biases -= self.lr * grad_biases
            
            # Store training loss
            loss = self._compute_loss(probabilities, y)
            self.train_loss_history.append(loss.item())
            
            # Validation loss and early stopping
            if X_val is not None and y_val is not None:
                val_loss = self._evaluate_loss(X_val, y_val)
                self.val_loss_history.append(val_loss)
                
                # early stopping
                if val_loss < best_val_loss - 1e-2:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        _print(f"Early stopping at epoch {epoch}")
                        self.epochs = epoch
                        break
    
    def plot_loss(self, train_loss_histories, val_loss_histories):
        """Plot mean and variance of training and validation loss curves."""
        # Pad the lists with NaNs to make them the same length
        max_length = max(max(len(hist) for hist in train_loss_histories), max(len(hist) for hist in val_loss_histories))
        train_loss_histories = [hist + [np.nan] * (max_length - len(hist)) for hist in train_loss_histories]
        val_loss_histories = [hist + [np.nan] * (max_length - len(hist)) for hist in val_loss_histories]
        train_loss_histories = np.array(train_loss_histories)
        val_loss_histories = np.array(val_loss_histories)

        mean_train_loss = np.mean(train_loss_histories, axis=0)
        std_train_loss = np.std(train_loss_histories, axis=0)

        mean_val_loss = np.mean(val_loss_histories, axis=0)
        std_val_loss = np.std(val_loss_histories, axis=0)

        fig, ax1 = plt.subplots()

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Train Loss', color='tab:blue')
        ax1.plot(range(len(mean_train_loss)), mean_train_loss, label='Train Loss', color='tab:blue')
        ax1.fill_between(range(len(mean_train_loss)), mean_train_loss - std_train_loss, mean_train_loss + std_train_loss, color='tab:blue', alpha=0.2)
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        if len(mean_val_loss) > 0:
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            ax2.set_ylabel('Validation Loss', color='tab:orange')
            ax2.plot(range(len(mean_val_loss)), mean_val_loss, label='Validation Loss', color='tab:orange')
            ax2.fill_between(range(len(mean_val_loss)), mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, color='tab:orange', alpha=0.2)
            ax2.tick_params(axis='y', labelcolor='tab:orange')

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title('Training and Validation Loss averaged across 5 cv folds')
        return plt
    
    def predict(self, X):
        """
        Predict class probabilities and labels for input data.
        """
        X = X.to(self.device)
        logits = self._compute_logits(X)
        probabilities = self._compute_probabilities(logits)
        predictions = torch.argmax(probabilities, dim=1)
        return probabilities.cpu().detach().numpy(), predictions.cpu().detach().numpy()

#%%
#--------# Grid search #--------#
model_type = "logisticregression"
results_prefix = args.dataset
param_grid = {
    "lambda_l1": [1e-4, 1e-3, 1e-2, 1e-1],
    "lr": [0.0002]
}

# Ensure CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_print(f"Using device: {device}")

_print(f"Training for seed {args.seednum}")

# Train test sets
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True, random_state = args.seednum)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_acc = 0
best_params = None
best_model = None
best_train_losses = None
best_val_losses = None

#--------# Model #--------#

for lambda_l1, lr in product(param_grid["lambda_l1"], param_grid["lr"]):
    fold_accuracies = []
    _print(f"Training model with lambda_l1={lambda_l1}, learning_rate={lr}")

    # Store loss histories for each fold
    train_loss_histories = []
    val_loss_histories = []

    for train_index, val_index in kf.split(X_trainval):

        X_train, X_val = X_trainval[train_index], X_trainval[val_index]
        y_train, y_val = y_trainval[train_index], y_trainval[val_index]

        model = MultinomialLogisticRegression(
            num_features=X_train.shape[1], 
            num_classes=5, 
            lr=lr, lambda_l1=lambda_l1, device=device
        )

        model.train(
            torch.tensor(X_train.toarray(), dtype=torch.float32), 
            torch.tensor(y_train, dtype=torch.long),
            torch.tensor(X_val.toarray(), dtype=torch.float32), 
            torch.tensor(y_val, dtype=torch.long)
        )

        probabilities, predictions = model.predict(torch.tensor(X_val.toarray(), dtype=torch.float32))
        acc = balanced_accuracy_score(y_val, predictions)
        fold_accuracies.append(acc)

        # Store loss histories
        train_loss_histories.append(model.train_loss_history)
        val_loss_histories.append(model.val_loss_history)

    avg_acc = np.mean(fold_accuracies)
    _print(f"Average Accuracy: {avg_acc}")

    if avg_acc > best_acc:
        best_acc = avg_acc
        best_params = (lambda_l1, lr)
        best_model = model
        best_train_losses = train_loss_histories
        best_val_losses = val_loss_histories

_print("End of grid search.\nBest Model Found:")
_print(f"Params: lambda_l1={best_params[0]}, learning_rate={best_params[1]}")
_print(f"Best Accuracy: {best_acc:.4f}")

model_name = f"{results_prefix}_{model_type}_{args.seednum}"
save_results(results_dir, model_name, best_model, X_train, X_test, y_train, y_test, best_train_losses, best_val_losses)
# %%