#!/usr/bin/env python3
#%%   
# dependencies
import gc
import os
import torch
import argparse
import numpy as np
import pandas as pd
import random
import warnings
from tqdm import tqdm
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torcheval.metrics.functional import multiclass_accuracy
from sklearn.model_selection import KFold

# import model library and training functions
import model_library
from model_library import train, test, evaluate, EarlyStopper

# import functions for plotting
from gnn_utils import plot_learning_curves, plot_model_metrics, _print, generate_intermediate_tsne_plot, initialize_logger, set_global_ngenes, AOGraphDataset

def main(model_name, seed_num, batch_size, learning_rate, epochs, hidden_channels, dataset, nconv):

    # model_name = 'GNNConvDropoutGlobalAttention'
    # seed_num = 42
    # batch_size = 8
    # learning_rate = 0.001
    # epochs = 30
    # hidden_channels = 64
    # dataset = 'binned_ao_GraphDataset01.pt'
    # nconv = 2

    ############## Directories ##############

    os.chdir('/pool01/projects/abante_lab/ao_prediction_enrollhd_2024/')
    # os.chdir('/gpfs/projects/ub212/ao_prediction_enrollhd_2024/data/')
    feat_dir = "features/"
    out_dir = "gnn/classification/"
    
    # Open log file to write execution outputs
    initialize_logger(out_dir)
    
    # model suffix with random string for uniqueness
    suff=f'{model_name}_nconv{nconv}_batch{batch_size}_lr{learning_rate}_epochs{epochs}_hl{hidden_channels}_dataset{dataset}_seed{seed_num}'
    _print(suff)
    _print("Start time")

    ############## Device to use ##############
    
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _print('Device:', device)

        # clear cuda cache
    torch.cuda.empty_cache()
    
    # Call garbage collector
    gc.collect()
    
    # # Query total GPU memory
    # device_id = 0  # Specify GPU index (there is only 1 in petra)
    # total_memory = torch.cuda.get_device_properties(device_id).total_memory  # In bytes

    # # Specify maximum memory in GB
    # max_memory_gb = 16
    # max_memory_fraction = max_memory_gb * 1024**3 / total_memory
    
    # # Or if we want to directly specify fraction of available gpu memory
    # max_memory_fraction = 0.5

    # # Set the memory fraction
    # torch.cuda.set_per_process_memory_fraction(max_memory_fraction, device=device_id)
    
    ############## Data loading ##############

    # Load .pt tensor without FutureWarning 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        dataset = torch.load(feat_dir + dataset)
    
    # Set random seed
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)
    random.seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)  # If using GPU

    # Shuffle the indices
    shuffled_indices = torch.randperm(len(dataset))

    # Split the shuffled dataset into train and testing datasets
    test_size = int(len(dataset) * 0.2)
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]
    test_dataset = Subset(dataset, test_indices)
    train_val_dataset = Subset(dataset, train_indices)

    # Create DataLoaders
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Set number of genes
    set_global_ngenes(ngenes=2774) # Get this from data

    _print('Data loaded.')
    
    
    ############## Model loading ##############
    
    # Parameters
    in_channels = dataset.gene_features_all.shape[2]
    
    # Dynamically load the model class
    ModelClass = getattr(model_library, model_name)

    # Optional learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    ############## Model training ##############

    # K-Fold Cross-Validation
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed_num)

    fold_results = []
    model = None
    
    # Initialize with the worst possible R²
    best_fold_acc = 0
    train_losses = []
    val_losses = []
    train_acc_scores = []
    val_acc_scores = []
    
    # Early stopping
    early_stopper = EarlyStopper(patience=20, min_delta=5e-4)

    for fold, (train_indices, val_indices) in enumerate(kf.split(train_val_dataset)):
        _print(f"\nStarting Fold {fold + 1}/{k_folds}")
        # fold = fold
        # train_indices = train_indices
        # val_indices = val_indices
        
        # Create Subsets for the current fold
        train_dataset = Subset(train_val_dataset, train_indices)
        val_dataset = Subset(train_val_dataset, val_indices)
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model, optimizer
        kmodel = ModelClass(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=nconv).to(device)
        optimizer = torch.optim.Adam(kmodel.parameters(), lr=learning_rate)

        # Metrics storage for this fold
        train_losses_k = []
        val_losses_k = []
        train_acc_scores_k = []
        val_acc_scores_k = []

        for epoch in range(1, epochs + 1):
        # for epoch in tqdm(range(1, epochs + 1), desc=f"Fold {fold + 1} Training Progress", ncols=100, unit="epoch"):
            # Train the model
            train_loss, train_true, train_pred = train(kmodel, optimizer, train_loader, device)
            train_losses_k.append(train_loss)

            # Compute accuracy for training
            train_acc = multiclass_accuracy(train_pred, train_true)
            train_acc_scores_k.append(train_acc)

            # Evaluate the model on validation data every 10 epochs
            if epoch % 10 == 0:
                val_loss, val_true, val_pred = test(kmodel, val_loader, device)
                val_losses_k.append(val_loss)

                # Compute accuracy for validation
                val_acc = multiclass_accuracy(val_pred, val_true)
                val_acc_scores_k.append(val_acc)

                _print(
                    f"\nFold: {fold + 1}, Epoch: {epoch:03d}, "
                    f"Train Loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val acc: {val_acc:.4f}"
                )
                
                if early_stopper.early_stop(val_loss):             
                    _print(f"Early stopping triggered at epoch {epoch} for fold {fold + 1}.")
                    break
                        
            else:
                val_losses_k.append(None)
                val_acc_scores_k.append(None)

        # Final evaluation for the fold
        _,_, acc = evaluate(kmodel, val_loader, device)
        fold_results.append(round(acc, 4))
        _print(f"Fold {fold + 1} Final Validation Accuracy: {acc:.4f}")

        # Check if this model is the best so far
        if acc > best_fold_acc:
            best_fold_acc = acc
            model = kmodel 
            train_losses = train_losses_k
            val_losses = val_losses_k
            train_acc_scores = train_acc_scores_k
            val_acc_scores = val_acc_scores_k
            # train_acc_scores = [t.item() for t in train_acc_scores_k]
            # val_acc_scores = [t.item() for t in val_acc_scores_k]

    _print(f'Fold acc: {fold_results}')
    _print(f'Using best model with acc: {best_fold_acc}')
    
    # Generate learning curves plot
    learning_fig = plot_learning_curves(train_losses, val_losses)
    learning_fig.savefig(out_dir+f'learning_curves_{suff}.png')
    # learning_fig.show()

    # Final evaluation on testing set
    test_y, test_yp, acc_test = evaluate(model, test_loader, device)
    
    # Check if accuracy is greater than 0.24
    if acc_test > 0.25:
        # Save the model and optimizer state_dict
        save_path = out_dir + f"model_and_optimizer_{suff}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'acc_score': acc_test,
        }, save_path)
        _print(f"Model saved successfully at {save_path} with accuracy: {acc_test}")
    else:
        _print(f"Model not saved. Accuracy ({acc_test}) did not exceed 0.24.")

    # Evaluate training set
    train__val_loader = DataLoader(train_val_dataset, batch_size=batch_size, shuffle=False)
    train_y, train_yp, _ = evaluate(model, train__val_loader, device)

    # Comparison true vs predicted labels
    metrics_summary, test_conf_mat, train_conf_mat = plot_model_metrics(train_y, train_yp, test_y, test_yp)
    # Save predicted vs true AO plot    
    test_conf_mat.savefig(out_dir+f'eval_fig_{suff}.png')
    train_conf_mat.savefig(out_dir+f'train_eval_fig_{suff}.png')

    _print(metrics_summary)
    # Save metrics in txt file
    with open(f'{out_dir}metrics_{suff}.txt', 'w') as f:
        f.write(metrics_summary + '\n')
        
    # t-sne of intermediate state colored by predicted and actual AO
    tsne = generate_intermediate_tsne_plot(model, test_loader, device)
    tsne.savefig(out_dir + f'intermediate_tsne_{suff}.png')
    
    # Save losses in dataframe
    data = {'train_losses': train_losses,
        'val_losses': val_losses,
        'train_acc_scores': train_acc_scores,
        'val_acc_scores': val_acc_scores,}
    loss_log = pd.DataFrame(data)

    # Save the DataFrame as a compressed .txt.gz
    loss_log.to_csv(out_dir + f'train_val_losses_acc_{suff}.txt.gz', sep='\t', index=False, compression='gzip')
    
    # clear cuda cache
    torch.cuda.empty_cache()
    
    # Call garbage collector
    gc.collect()

    _print('End time.')
    
if __name__ == "__main__":
    
    # argument parser
    parser = argparse.ArgumentParser(description='Train a GNN model.')
    
    # mandatory arguments
    parser.add_argument('--model', type=str, required=True, help='Name of the model class to use from model_library.py')
    
    # optional arguments
    parser.add_argument('--seed', type=int, required=False, default=42, help='Random seed')
    parser.add_argument('--batch', type=int, required=False, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, required=False, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, required=False, default=500, help='Number of epochs')
    parser.add_argument('--hl', type=int, required=False, default=64, help='Number of hidden channels')
    parser.add_argument('--dataset', type=str, required=False, default='binned_ao_GraphDataset01.pt', help='Dataset')
    parser.add_argument('--nconv', type=int, required=False, default=2, help='Number of convolutional layers')

    # parse args
    args = parser.parse_args()
    
    # call main 
    main(args.model, seed_num=args.seed, batch_size=args.batch, learning_rate=args.lr, epochs=args.epochs, hidden_channels=args.hl, dataset=args.dataset, nconv=args.nconv)
# %%
