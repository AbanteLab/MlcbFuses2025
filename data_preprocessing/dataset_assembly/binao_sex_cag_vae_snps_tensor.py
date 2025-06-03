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
with open(feat_dir+'gene_tensor_order_nonrep.txt') as f:
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

# AO residuals version of y_all_unn
y_all_unn = pd.read_csv(feat_dir + 'binned_ao.txt', header = None)
y_all_unn = y_all_unn[0].values
y_all_unn = torch.tensor(y_all_unn, dtype=torch.long)

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
    
dataset = AOGraphDataset(scaled_features, sex_all, cag_normalized, y_all_unn, edge_index, edge_attr)

torch.save(dataset, feat_dir+'binned_ao_GraphDataset01.pt')
# %%
