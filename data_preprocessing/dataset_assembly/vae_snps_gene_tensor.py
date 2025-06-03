#!/usr/bin/env python3

### DATA PREPARATION FOR GNN INPUT ###

import numpy as np 
import pandas as pd
import os
from scipy.sparse import csr_matrix, vstack
import torch
import re
from datetime import datetime

def _print(*args, **kw):
    print("[%s]" % (datetime.now()),*args, **kw)
    
_print('Starting time.')

os.chdir('/pool01/projects/abante_lab/ao_prediction_enrollhd_2024/')

embedding_dir = 'backup_MN5/data/enroll_hd/vae_embeddings'
feature_dir = 'features/'
feature_matrix_path = f'{feature_dir}feature_matrix_m3_filt_0.01.txt'
# header_feature_matrix_path  = 'data/features/subsetting/header_feature_matrix_m3_filt_0.01.txt'
z_dims_path = f'{feature_dir}zdims_genes.txt'
gene_SNP_path = 'biomart/revised_filtered_snp_gene_lookup_tab.txt'


def read_sparse_X(X_path, chunk_size=100):
    '''Reads tab separated matrix stored in X_path by 
    row groups of size chunk_size, transforms into csr matrix
    with data type np.float32 and concatenates all.'''

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
    
    return X, header

# Read feature matrix
snp_feat_matrix, header = read_sparse_X(feature_matrix_path)

_print('Feature matrix loaded')

# Save Sex and CAG separately
sex_CAG =  snp_feat_matrix[:,0:2].toarray()
np.save(f'{feature_dir}sexCAG.npy', sex_CAG)

snp_feat_matrix = snp_feat_matrix[:,2:]
header = header[3:]

# Get maximum value of snp matrix
snp_max = snp_feat_matrix.max()

# Read lookup table 
gene_snps = pd.read_csv(gene_SNP_path, sep='\t')

# Read table with #SNPs and #LD blocks (zdim in VAE)
zdims = pd.read_csv(z_dims_path, sep='\t', names = ['gene', 'zdim', 'n_snps'], header=None)

# Filter genes that have less than 30 SNPs
genes_feat_snps = zdims[zdims['n_snps']<=30]

# Filter genes that have more than 30 SNPs -> VAE embeddings
genes_feat_vae = zdims[zdims['n_snps']>30]

_print('{} genes with SNPs as features, {} genes with VAE embeddings as features'.format(len(genes_feat_snps), len(genes_feat_vae)))

# Genes with VAE embeddings

gene_tensors = []
gene_order = []

# Loop through each file in the gene directory
for gene_file in sorted(os.listdir(embedding_dir)):
    file_path = os.path.join(embedding_dir, gene_file)
    if os.path.isfile(file_path) and gene_file.endswith('.txt.gz'):
        # Extract the gene name using regex
        match = re.match(r'embeddings_(.+)\.txt\.gz', gene_file)
        if match:
            gene_name = match.group(1)
            
            # Load each gene file into a pandas DataFrame
            gene_data = pd.read_csv(file_path, sep = '\t', header=None, compression='gzip') 
            
            # Convert DataFrame to a torch tensor
            gene_tensor = torch.tensor(gene_data.values, dtype=torch.float32)
            
            # Zero-pad the gene tensor to shape (samples, 30)
            padded_gene_tensor = torch.nn.functional.pad(gene_tensor, (0, 30 - gene_tensor.shape[1]))
            
            # Normalize the tensor to [0, 1]
            min_val = padded_gene_tensor.min()
            max_val = padded_gene_tensor.max()
            normalized_tensor = (padded_gene_tensor - min_val) / (max_val - min_val + 1e-8)  # Add epsilon to prevent division by zero
            
            # Scale the tensor to [0, snp_max] tp have same range as snp coded genes
            scaled_tensor = normalized_tensor * snp_max
            
            # Add the scaled tensor to the list
            gene_tensors.append(scaled_tensor)

            # Append the gene name to the order list, as many times as columns added as features
            for _ in range(gene_tensor.shape[1]):
                gene_order.append(gene_name)

# See if all genes with > 30 SNPs are in current tensor
_print('control 1:', set(gene_order) == set(genes_feat_vae['gene']))
_print('VAE tensors created')

# Add to tensor SNPs genes
# Loop through each gene in your list
for gene in genes_feat_snps['gene']:
    
    # Filter the DataFrame for the current gene
    gene_data = gene_snps[gene_snps['gene'] == gene]

    # Get the refsnp_id for the current gene
    refsnp_ids = gene_data['refsnp_id'].values

    # Find the positions of these refsnp_ids in the header
    positions = []
    for refsnp_id in refsnp_ids:
        if refsnp_id in header:
            positions.append(header.index(refsnp_id))
            
    # Slice the matrix to get the data for these refsnp_ids
    gene_matrix = snp_feat_matrix[:, positions].toarray()

    # Convert the sliced matrix to a PyTorch tensor
    gene_tensor = torch.tensor(gene_matrix, dtype=torch.float32)

    # Zero-pad the tensor to have 30 columns
    padded_gene_tensor = torch.nn.functional.pad(gene_tensor, (0, 30 - gene_tensor.shape[1]))

    # Add the padded gene tensor to the list
    gene_tensors.append(padded_gene_tensor)

    # Append the gene name to the order list, as many times as columns added as features
    for _ in range(len(gene_data)):
        gene_order.append(gene)

_print('SNPs tensors created')

# Stack the list of gene tensors along a new dimension to create the final tensor
# The resulting shape will be (samples, genes, 30)
final_tensor = torch.stack(gene_tensors, dim=1)

torch.save(final_tensor, f'{feature_dir}gene_tensor.pth')

# Save the gene order to a text file
with open(f'{feature_dir}gene_tensor_order.txt', 'w') as f:
    for gene_name in gene_order:
        f.write(gene_name + '\n')
        
_print('Final tensor of shape {} saved.'.format(final_tensor.shape))