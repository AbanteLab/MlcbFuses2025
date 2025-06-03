#!/usr/bin/env python3

### Dataset object creation with SNPs+VAE embeddings and PPI interactions ###
#%%
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
import numpy as np
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, lil_matrix
from sklearn.preprocessing import MinMaxScaler

os.chdir('/pool01/projects/abante_lab/ao_prediction_enrollhd_2024/')

ppi_dir = "ppi_networks/"
feat_dir = "features/"

# Node feature matrix for all samples
gene_features_all = torch.load(feat_dir + 'gene_tensor.pth', weights_only=False) 

# Load adjacency matrix (scipy LIL matrix)
with open(ppi_dir + f"protein_interactions_matrix_total.pkl", "rb") as file:
    adj_matrix = pickle.load(file)
    
# Load the order of genes in adjacency matrix
with open(ppi_dir+'gene_order_total.txt') as f:
    adj_gene_order = [line.strip() for line in f]

# Load the order of genes in gene tensor
with open(feat_dir+'gene_tensor_order.txt') as f:
    features_gene_order = [line.strip() for line in f]

#%% For features to be between 0 and 1

# Compute min and max for each feature across the last dimension (size 30)
min_vals = torch.amin(gene_features_all, dim=(0, 1), keepdim=True)  # Shape: [1, 1, 30]
max_vals = torch.amax(gene_features_all, dim=(0, 1), keepdim=True)  # Shape: [1, 1, 30]

# Apply Min-Max Scaling
scaled_features = (gene_features_all - min_vals) / (max_vals - min_vals)

# To handle potential divide-by-zero if min == max for any feature:
scaled_features = torch.nan_to_num(scaled_features, nan=0.0)

#%% Reorder the adjacency matrix

# Create a mapping from adj_gene_order to features_gene_order
gene_to_index = {gene: idx for idx, gene in enumerate(features_gene_order)}

# Convert to COO format if not already
coo = adj_matrix.tocoo()

# Map adjacency genes to their new order
new_order = [gene_to_index[gene] for gene in adj_gene_order]

# Remap the rows, columns, and data directly
row = np.array([new_order[idx] for idx in coo.row])
col = np.array([new_order[idx] for idx in coo.col])
data = coo.data

# Reshape values to 2D (required by sklearn)
data_2d = data.reshape(-1, 1)

# Apply MinMaxScaler to have edge attr between 0 and 1 
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_edge_attr = scaler.fit_transform(data_2d).flatten()

# Create a new COO matrix using reordered rows, columns, and data
reordered_adj_matrix = coo_matrix((scaled_edge_attr, (row, col)), shape=adj_matrix.shape)

# Create edge index and edge attributes directly
edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)
edge_attr = torch.tensor(scaled_edge_attr, dtype=torch.float32)

#%% Sex and CAG vectors
sexCAG = np.load(feat_dir + 'sexCAG.npy')
sex_all = torch.tensor(sexCAG[:,0], dtype=torch.float32)
cag_all = torch.tensor(sexCAG[:,1], dtype=torch.float32)

# Min-max CAG scaling
cag_min = torch.min(cag_all)
cag_max = torch.max(cag_all)
cag_normalized = (cag_all - cag_min) / (cag_max - cag_min)

# Map 1. -> 0, 2. -> 1 the sex vector
sex_all = (sex_all - 1).long()

#%% AO values
# y_all_unn = pd.read_csv(feat_dir + 'aoo.txt', sep='\t')
# y_all_unn = y_all_unn['Onset.Age'].values

# AO residuals version of y_all_unn
y_all_unn = pd.read_csv(feat_dir + 'AO_residuals.txt', header = None)
y_all_unn = y_all_unn[0].values

y_mean = y_all_unn.mean()
y_std = y_all_unn.std()

# Standardize target
y_all = (y_all_unn - y_mean) / y_std

class AOGraphDataset(Dataset):
    def __init__(self, gene_features_all, sex_all, cag_all, y_all, edge_index, edge_attr, mean, std):
        self.gene_features_all = gene_features_all
        self.sex_all = sex_all
        self.cag_all = cag_all
        self.y_all = y_all
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.mean = mean
        self.std = std

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
#%% features scaled between 0 and 1, and y values standardized
# dataset = AOGraphDataset(scaled_features, sex_all, cag_normalized, y_all, edge_index, edge_attr, y_mean, y_std)

# torch.save(dataset, feat_dir+'AOGraphDataset01.pt')

#%% For features to be between 0 and 1 and AO residuals, standardized
dataset = AOGraphDataset(scaled_features, sex_all, cag_normalized, y_all, edge_index, edge_attr, y_mean, y_std)

torch.save(dataset, feat_dir+'AO_residuals_GraphDataset01.pt')
#%%
# Permute adjacency matrix
def permute_adjacency_matrix(adj_matrix, percentage, seed=None):
    """
    Permutes a given percentage of edges in the adjacency matrix.

    Parameters:
    - adj_matrix (scipy.sparse.coo_matrix): Original adjacency matrix in COO format.
    - percentage (float): Percentage of edges to permute (0 to 1).
    - seed (int): Random seed for reproducibility.

    Returns:
    - new_adj_matrix (scipy.sparse.coo_matrix): Permuted adjacency matrix.
    """
    if seed is not None:
        np.random.seed(seed)

    # Get rows, columns, and data from the COO format
    rows, cols, data = adj_matrix.row, adj_matrix.col, adj_matrix.data
    num_edges = len(data)
    
    # Number of edges to permute
    num_permute = int(percentage * num_edges)
    
    # Randomly select edges to permute
    permute_indices = np.random.choice(num_edges, num_permute, replace=False)
    
    # Generate new edges
    new_rows = rows.copy()
    new_cols = cols.copy()
    
    for idx in permute_indices:
        # Remove the old edge
        new_rows[idx] = np.random.randint(0, adj_matrix.shape[0])
        new_cols[idx] = np.random.randint(0, adj_matrix.shape[1])
        
        # Ensure no self-loops and no duplicate edges
        while new_rows[idx] == new_cols[idx] or (new_rows[idx], new_cols[idx]) in zip(rows, cols):
            new_rows[idx] = np.random.randint(0, adj_matrix.shape[0])
            new_cols[idx] = np.random.randint(0, adj_matrix.shape[1])

    # Create a new adjacency matrix
    new_adj_matrix = coo_matrix((data, (new_rows, new_cols)), shape=adj_matrix.shape)

    return new_adj_matrix

# Apply permutation to the adjacency matrix
percentage_to_permute = 0.1  # 10% of edges
seed = 42  # Set random seed for reproducibility
new_adj_matrix = permute_adjacency_matrix(reordered_adj_matrix, percentage_to_permute, seed)

# Convert the new adjacency matrix to edge_index and edge_attr
new_coo = new_adj_matrix.tocoo()
new_edge_index = torch.tensor(np.array([new_coo.row, new_coo.col]), dtype=torch.long)
new_edge_attr = torch.tensor(new_coo.data, dtype=torch.float32)

dataset = AOGraphDataset(gene_features_all, sex_all, cag_normalized, y_all, new_edge_index, new_edge_attr, y_mean, y_std)

torch.save(dataset, feat_dir+'AOGraphDataset_permutation10.pt')