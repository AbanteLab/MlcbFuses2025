#!/usr/bin/env python3

### XGBoost classification ###
#%%
import os
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import balanced_accuracy_score
from itertools import product
import xgboost as xgb
import sys
import argparse

PROJECT_ROOT = "/pool01/code/projects/abante_lab/ao_prediction_enrollhd_2024/ml_models"
# PROJECT_ROOT = "/gpfs/projects/ub212/ao_prediction_enrollhd_2024/code/src/ml_models"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load custom functions
from utils.evaluating_functions import evaluate_model
from utils.data_loading import _print, load_X_y

# Parse command-line arguments
parser = argparse.ArgumentParser(description="XGBoost Classification Script")
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
#--------# Grid search #--------#
model_type = "ES_xgboost"
results_prefix = args.dataset
param_grid = {
        'max_depth': [1, 2, 3, 6],
        'learning_rate': [0.001, 0.01, 0.1]
    }
# Early stopping parameters
early_stopping_rounds = 10

# Train test and validation sets
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True, random_state = args.seednum)

kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-Fold CV like Method 1
best_acc = 0
best_params = None
best_model = None

#--------# Model #--------#

for max_depth, learning_rate in product(param_grid["max_depth"], param_grid["learning_rate"]):
    fold_accuracies = []
    _print(f"Training XGBoost with max_depth={max_depth}, learning_rate={learning_rate}")

    for train_index, val_index in kf.split(X_trainval):
        X_train, X_val = X_trainval[train_index], X_trainval[val_index]
        y_train, y_val = y_trainval[train_index], y_trainval[val_index]

        # Convert to DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val)

        # Use XGBoost CV to determine the best number of boosting rounds (early stopping)
        cv_results = xgb.cv({"random_state": 42, "max_depth": max_depth, "learning_rate": learning_rate,
                             "tree_method": "hist", "device": "cuda", "num_class": 5,
                             "objective": "multi:softmax", "eval_metric": 'mlogloss', "verbosity": 2},
                            dtrain, num_boost_round=1000, nfold=5,  # Internal 5-fold CV in XGBoost
                            early_stopping_rounds=10, as_pandas=True)

        # Extract best num_boost_rounds
        best_boosting_rounds = cv_results.shape[0]

        # Train final model
        model = xgb.train({"random_state": 42, "max_depth": max_depth, "learning_rate": learning_rate,
            "tree_method": "hist", "device": "cuda", "num_class": 5,
            "objective": "multi:softmax", "eval_metric": 'mlogloss', "verbosity": 2},
            dtrain, num_boost_round=best_boosting_rounds)

        # Predict on validation set
        y_pred = model.predict(dval)

        # Compute accuracy
        acc = balanced_accuracy_score(y_val, y_pred)
        fold_accuracies.append(acc)
    
    avg_acc = np.mean(fold_accuracies)
    _print(f"Average Accuracy: {avg_acc:.4f}")

    # Track Best Model
    if avg_acc > best_acc:
        best_acc = avg_acc
        best_params = (max_depth, learning_rate)
        best_model = model

_print("\nBest Model Found:")
_print(f"Params: max_depth={best_params[0]}, learning_rate={best_params[1]}")
_print(f"Best Accuracy: {best_acc:.4f}")

# Evaluate on test set
acc, train_acc, string_mets, train_cm, test_cm, conf_matx_plot = evaluate_model(best_model, X_train, X_test, y_train, y_test)

model_name = f"{results_prefix}_{model_type}_{args.seednum}"

with open(f"{results_dir}{model_name}_metrics.txt", 'w') as file:
    file.write(f'Model name: {model_name}\n')
    file.write(f'Model balanced accuracy: {acc}\n')
    file.write(f'Model balanced train accuracy: {train_acc}\n')
    file.write('\nModel report:\n')
    for key, value in string_mets.items():
        file.write(f'{key}: {value}\n')
    file.write('\nModel parameters:\n')
    file.write(str(best_model.save_config()))
    file.write('\nTraining confusion matrix:\n')
    file.write(str(train_cm))
    file.write('\nTest confusion matrix:\n')
    file.write(str(test_cm))

conf_matx_plot.savefig(f"{results_dir}{model_name}_test_confmatx.png")

# Save the model
best_model.save_model(f"{results_dir}{model_name}_model.json")

trees = best_model.trees_to_dataframe()
trees.to_csv(f"{results_dir}{model_name}_trees.csv", sep="\t")

_print("Results saved.")