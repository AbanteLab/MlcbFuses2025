#!/usr/bin/env python3

### Custom functions for VAE training setup ###

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, DataLoader, TensorDataset
import pyro
import pyro.distributions as dist
from pyro import poutine
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from datetime import datetime

def _print(*args, **kw):
    print("[%s]" % (datetime.now()),*args, **kw)

def read_sparse_X(X_path, chunk_size=100):
    '''Reads tab separated matrix stored in X_path by 
    row groups of size chunk_size, transforms into csr matrix
    with data type np.float32 and concatenates all.'''

    from scipy.sparse import csr_matrix, vstack
    import numpy as np

    # Initialize an empty list to store the chunks
    chunks = []    
        
    # Open the file
    with open(X_path, 'r') as file:
        # Read header line
        header = file.readline().strip().split("\t")

        # Initialize a list to store chunk data
        chunk_data = []

        # Read the file in chunks
        while True:
            # Read chunk_size lines
            for _ in range(chunk_size):
                
                line = file.readline()
                
                if not line:
                    break  # Reached end of line
                
                data = line.strip().split("\t")
                
                # Add feature data to row vector as float32
                rowdata = [np.float32(val) for val in data[1:]]
                
                # Add row vector to chunk_data list
                chunk_data.append(rowdata)

            if not chunk_data:
                break  # No more data to read

            # Convert the chunk data to a CSR matrix
            chunk_sparse = csr_matrix(chunk_data)

            # Append the chunk to the list
            chunks.append(chunk_sparse)

            # Clear chunk data for the next iteration
            chunk_data = []
            
    # Concatenate the list of CSR matrices into a single CSR matrix
    X = vstack(chunks)
    
    return X

def import_data(X_path, lookuptab_path, header_path,):
    '''Reads necessary files to assemble gene_matrix.'''
    
    # Load X matrix
    X = read_sparse_X(X_path, chunk_size = 100)

    # Load lookup table
    lookuptab = pd.read_csv(lookuptab_path, sep = "\t") 

    # Load header
    with open(header_path, 'r') as file:
        line = file.readline()
    header = line.strip().split('\t')

    # Leave out first element of header (FID_IID) because this column is not in X
    header = header[1:]

    return X, lookuptab, header

def load_gene_matrix(X, lookuptab, header, gene):
    '''Subsets the feature matrix X taking only the SNPs from the 
    input gene.'''
    
    # Get IDs of all SNPs from gene
    gene_snps = lookuptab.loc[lookuptab["gene"] == gene, "refsnp_id"].to_list()

    # Subset only those contained in feature matrix
    gene_snps = set(gene_snps).intersection(header)
    
    gene_snps_idxs = []

    # Iterate over the list with enumerate to get both index and value
    for index, value in enumerate(header):
        if value in gene_snps:
            gene_snps_idxs.append(index)

    # Subset feature matrix
    gene_matrix = X[:,gene_snps_idxs].toarray()
    
    return gene_matrix

# Class to handle data in torch
class GeneMatrixDataset(Dataset):

    def __init__(self, gene_matrix):
        self.gene_matrix = gene_matrix

    def __len__(self):
        return len(self.gene_matrix)

    def __getitem__(self, idx):
        sample = self.gene_matrix[idx]
        return torch.tensor(sample, dtype=torch.float32)
    
def save_indices_to_txt(indices, filename):
    """Helper function to save indices to a text file."""
    with open(filename, 'w') as f:
        for index in indices:
            f.write(f"{index}\n")    

# Function to setup data loaders
def setup_data_loaders(train_indices_path, test_indices_path, gene_matrix, 
                       batch_size=128, use_cuda=False, train_split=0.9,save_indices=True):
    
    np.random.seed(25)
    torch.manual_seed(25)
    
    # Create a dataset instance
    dataset = GeneMatrixDataset(gene_matrix)

    # Calculate the number of samples for training and testing
    total_size = len(dataset)
    train_size = int(total_size * train_split)
    test_size = total_size - train_size
    
    # Get all indices and shuffle them
    indices = list(range(total_size))
    np.random.shuffle(indices)

    # Split the indices into training and testing subsets
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Create Subset datasets
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    # Define data loaders
    kwargs = {'num_workers': 1, 'pin_memory': use_cuda}
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    
    # Save indices
    if save_indices:
        save_indices_to_txt(train_indices, train_indices_path)
        save_indices_to_txt(test_indices, test_indices_path)
    
    return train_loader, test_loader

# Define training loop
def train(svi, train_loader, use_cuda=False):
    # initialize loss accumulator
    epoch_loss = 0.
    # do a training epoch over each mini-batch x returned
    # by the data loader
    for x in train_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
        # do ELBO gradient and accumulate loss
        epoch_loss += svi.step(x)

    # return epoch loss
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train

def evaluate(svi, test_loader, use_cuda=False):
    # initialize loss accumulator
    test_loss = 0.
    # compute the loss over the entire test set
    for x in test_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
        # compute ELBO estimate and accumulate loss
        test_loss += svi.evaluate_loss(x)  # doesn't take any gradient steps
    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test

def custom_evaluate_loss(model, guide, data_loader):
    
    reconstruction_loss_total = 0.0
    kl_divergence_total = 0.0
    
    for data in data_loader:
        # run the guide and replay the model against the guide
        guide_trace = poutine.trace(lambda x: guide(x)).get_trace(data)
        model_trace = poutine.trace(
            poutine.replay(lambda x: model(x), trace=guide_trace)).get_trace(data)

        reconstruction_loss = 0.0
        kl_divergence = 0.0
        
        # loop through all the sample sites in the model and guide trace and
        # construct the loss
        for name, site in model_trace.nodes.items():
            if site["type"] == "sample":
                reconstruction_loss_t = site["fn"].log_prob(site["value"]).sum()
                reconstruction_loss += reconstruction_loss_t.detach().cpu().numpy().item()
    
        for name, site in guide_trace.nodes.items():
            if site["type"] == "sample":
                kl_divergence_t = site["fn"].log_prob(site["value"]).sum()
                kl_divergence += kl_divergence_t.detach().cpu().numpy().item()
    
        # Aggregate losses across batches and normalize by batch size
        reconstruction_loss_total += reconstruction_loss
        kl_divergence_total += kl_divergence
        
    # Return average loss over all batches
    normalizer_test = len(data_loader.dataset)
    
    reconstruction_loss_total = reconstruction_loss_total/normalizer_test
    kl_divergence_total = kl_divergence_total/normalizer_test
    
    elbo_total = reconstruction_loss_total - kl_divergence_total
    
    return elbo_total, reconstruction_loss_total, kl_divergence_total

def plot_test_elbo(test_elbo, num_epochs, test_frequency):
    '''Create a plot of test ELBO over epochs and return the figure object.'''
    
    # Create epoch range for the x-axis
    epochs = range(0, num_epochs, test_frequency)
    
    # Create a figure and axis object
    fig, ax = plt.subplots()
    
    # Plot the data
    ax.plot(epochs, test_elbo, marker='o', linestyle='-', color='b', label='Test ELBO')
    
    # Set labels and title
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Test ELBO')
    ax.set_title('Test ELBO over Epochs')
    
    # Add grid and legend
    ax.grid(True)
    ax.legend()
    
    # Return the figure object
    return fig

def plot_losses(reconstruction_losses, kl_divergences, total_losses, num_epochs, test_frequency):
    '''Plots separately the reconstruction loss and the KL divergence.'''
    
    # Create epoch range for the x-axis
    epochs = range(0, num_epochs, test_frequency)
    
    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot Reconstruction Loss
    ax.plot(epochs, reconstruction_losses, color='green', label='Reconstruction Loss')
    
    # Plot KL Divergence
    ax.plot(epochs, kl_divergences, color='red', label='KL Divergence')
    
    # Plot elbo
    ax.plot(epochs, total_losses, color='red', label='Total loss')
    
    # Set labels and title
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Reconstruction Loss and KL Divergence Over Epochs')
    
    # Add legend
    ax.legend()
    
    ax.grid(True)
    
    # Return the figure object
    return fig

def encode_data(vae, gene_matrix, use_cuda=False):
    '''Takes SNP matrix of a certain gene and a trained VAE, and
    encodes the matrix using the already trained encoder. Saves only the mean.'''
    # Set the VAE to evaluation mode
    vae.eval()  
    
    # Create a DataLoader for the entire dataset
    dataset = GeneMatrixDataset(gene_matrix)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    encoded_data = []

    with torch.no_grad():  # Disable gradient computation
        for batch in data_loader:
            if use_cuda:
                batch = batch.cuda()
            
            # Pass through the encoder to get latent representations
            z_loc, _ = vae.encoder(batch)
            encoded_data.append(z_loc.cpu().numpy())  # Append encoded data to list
    
    # Concatenate all batches of encoded data
    encoded_data = np.concatenate(encoded_data, axis=0)
    
    return encoded_data

def decode_data(vae, encoded_data, batch_size=128, f_dec = "Bernoulli", use_cuda=False):
    vae.eval()  # Set the VAE to evaluation mode
    
    # Create a DataLoader for the encoded data
    encoded_dataset = TensorDataset(torch.tensor(encoded_data, dtype=torch.float32))
    data_loader = DataLoader(encoded_dataset, batch_size=batch_size, shuffle=False)
    
    decoded_data = []

    with torch.no_grad():  # Disable gradient computation
        for batch in data_loader:
            batch = batch[0]  # Get the tensor from the TensorDataset
            if use_cuda:
                batch = batch.cuda()
            
            if f_dec == "Bernoulli":
                # Pass through the decoder to get reconstructed data
                decoded_batch = vae.decoder(batch)
                
            elif f_dec == "Categorical":
                # Pass through the decoder to get logits for the categorical distribution
                logits = vae.decoder(batch)
                
                # Convert logits to probabilities
                probs = nn.Softmax()(logits)
                
                # Get the most probable category (argmax of probabilities)
                decoded_batch = torch.argmax(probs, dim=-1)
            
            # Append decoded data to list, passing to cpu
            decoded_data.append(decoded_batch.cpu().numpy())  
    
    # Concatenate all batches of decoded data
    decoded_data = np.concatenate(decoded_data, axis=0)
    
    return decoded_data

def running_mean(arr, window_size):
    '''Calculate the running mean for each row of array arr.'''
    
    cumsum = np.cumsum(arr, dtype=float)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    
    return cumsum[window_size - 1:] / window_size

def gaussian_smoothing(x, sigma):
    '''Smooth a 1D array x using a Gaussian filter with standard deviation sigma.'''
    return gaussian_filter1d(x, sigma)

def snps_latent_correlation(encoded_data, gene_matrix, sigma_percentage = 0.03):
    '''Computes the correlation of each latent variable with the input SNPs and
    represents it in two plots, one (fig) showing the raw correlations individually for each
    latent variable z, and another (fig2) showing the running mean in a single set of axes.'''
    
    # Create an empty matrix to store the correlations
    correlation_matrix = np.empty((encoded_data.shape[1], gene_matrix.shape[1]))

    # Compute the correlation for each column in encoded_matrix
    for i in range(encoded_data.shape[1]):
        encoded_col = encoded_data[:, i]
        for j in range(gene_matrix.shape[1]):
            gene_col = gene_matrix[:, j]
            # Compute Pearson correlation coefficient
            correlation = np.corrcoef(encoded_col, gene_col)[0, 1]
            correlation_matrix[i, j] = correlation
            
    z_snps_corr = abs(correlation_matrix)
    
    # Number of rows in the matrix to see
    num_rows = z_snps_corr.shape[0]

    # Create subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=1, figsize=(10, 2*num_rows), sharex=True, sharey=True)

    # Plot each row in a subplot
    for i in range(num_rows):
        axes[i].plot(z_snps_corr[i, :])
        axes[i].set_title(f'z{i+1}')
        axes[i].grid(True)

    # Add a single x-axis label
    fig.text(0.5, 0.005, 'SNP', ha='center', va='center', fontsize=12)
    
    # Add title
    fig.suptitle("Correlation between latent variables and input SNPs", fontsize=16)
    
    plt.tight_layout()
    
    # Single plot
    fig2, ax2 = plt.subplots(figsize=(15, 6))
    
    sigma = int(round(gene_matrix.shape[1] * sigma_percentage, 0))
    
    # Plot each row's running mean
    for i in range(z_snps_corr.shape[0]):
        z_smoothed = gaussian_smoothing(z_snps_corr[i, :], sigma)
        ax2.plot(z_smoothed, label=f'z{i+1}')
    
    # Add labels, title, legend, and grid
    ax2.set_xlabel('SNP Index')
    ax2.set_ylabel('Smoothed Correlation')
    ax2.set_title('Correlation z-SNPs')
    ax2.legend()
    ax2.grid(True)
    
    # Return the figure object
    return fig, fig2

def latent_correlations(encoded_data):
    '''Computes the correlation between the latent variables and produces a heatmap.'''

    # Create a DataFrame
    encoded_df = pd.DataFrame(encoded_data)

    # Compute the correlation matrix
    encoded_corr_matrix = encoded_df.corr()
    
    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create the heatmap
    cax = ax.imshow(abs(encoded_corr_matrix), cmap='OrRd', vmin=0, vmax=1)

    # Add a colorbar
    cbar = fig.colorbar(cax, ax=ax, shrink=0.8)

    # Set the title
    ax.set_title('Zs Absolute Correlation Coefficient Matrix')

    # Set x and y ticks dynamically based on the number of columns
    num_labels = encoded_corr_matrix.shape[1]
    tick_positions = np.arange(num_labels)
    labels = [f'z{i+1}' for i in tick_positions]

    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    # Return the figure object
    return fig

def bin_reconstructed(decoded_data, args):
    '''Bins the values of the numpy array 'decoded_data' into 0, 0.5 and 1, both for 
    Bernoulli and Categorical cases.'''
    
    # Define the bin edges and bin labels
    bins = np.array([0, 1/3, 2/3, 1])  # Bin edges
    bin_labels = np.array([0, 0.5, 1])  # Corresponding bin labels
    
    if args.f_decoder == "Categorical":
        # Duplicate the values of bin edges to span from 0 to 2
        bins = 2*bins        

    # Flatten the matrix for vectorized operations
    flattened_data = decoded_data.flatten()

    # Use numpy to find the bin indices for each value
    # Ensure that values exactly equal to the upper edge of the last bin get the last label
    bin_indices = np.searchsorted(bins, flattened_data, side='right') - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_labels) - 1)

    # Map bin indices to bin labels
    binned_flattened_data = bin_labels[bin_indices]

    # Reshape back to the original matrix shape
    binned_matrix = binned_flattened_data.reshape(decoded_data.shape)

    return binned_matrix

def matrix_similarity(A, B, args):
    '''Compares two arrays by calculating the difference A-B and:
    - Computing the norm of each column (SNP) and summing all norms.
    - Heatmap of the difference matrix.'''
    
    # Ensure A and B are numpy arrays
    A = np.array(A)
    B = np.array(B)
    
    # Difference matrix
    diff = A - B
    
    # Normalize to be comparable to Bernoulli results
    if args.f_decoder == "Categorical":
        diff = diff * 0.5

    # Row norms sum
    row_norm = np.mean(np.linalg.norm(diff, axis=1))
    
    # Define a custom colormap from -1 to 1
    cmap = plt.get_cmap('coolwarm')
    norm = mcolors.Normalize(vmin=-1, vmax=1)

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(diff, cmap=cmap, norm=norm, aspect='auto')
    fig.colorbar(cax, label='Difference real - reconstructed')

    # Add labels and title for better understanding
    ax.set_title('Reconstruction difference {}'.format(args.gene))
    ax.set_xlabel('SNP')
    ax.set_ylabel('Sample')

    plt.tight_layout()
    
    return row_norm, fig

def confusion_matrix(A, B, args):
    '''Confusion matrix between original matrix A and the reconstructed matrix B.
    B matrix should have binned elements.'''

    # Initialize the confusion matrix
    confusion_matrix = np.zeros((3, 3))

    # Normalize to be comparable to Bernoulli results
    if args.f_decoder == "Categorical":
        A = A * 0.5
        B = B * 0.5

    # Fill the confusion matrix
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            a_val = int(A[i, j] * 2)  # Convert 0, 0.5, 1 to 0, 1, 2
            b_val = int(B[i, j] * 2)  # Convert 0, 0.5, 1 to 0, 1, 2
            confusion_matrix[a_val, b_val] += 1

    # Calculate accuracy at reconstructing each genotype
    acc0 = confusion_matrix[0,0]/sum(confusion_matrix[0])
    acc1 = confusion_matrix[1,1]/sum(confusion_matrix[1])
    acc2 = confusion_matrix[2,2]/sum(confusion_matrix[2])

    # Create the figure and axis
    fig, axs = plt.subplots()

    # Plot the confusion matrix
    cax = axs.imshow(confusion_matrix, cmap='Blues', interpolation='nearest')
    fig.colorbar(cax, ax=axs)

    # Set ticks and labels
    axs.set_xticks([0, 1, 2])
    axs.set_yticks([0, 1, 2])
    axs.set_xticklabels(['0', '0.5', '1'])
    axs.set_yticklabels(['0', '0.5', '1'])
    axs.set_xlabel('Reconstructed gene matrix')
    axs.set_ylabel('Input gene matrix')
    axs.set_title('Confusion Matrix {}'.format(args.gene))

    # Annotate each cell with the numeric value
    for i in range(3):
        for j in range(3):
            axs.text(j, i, int(confusion_matrix[i, j]), ha='center', va='center', color='black')

    # Return the figure object
    return fig, acc0, acc1, acc2