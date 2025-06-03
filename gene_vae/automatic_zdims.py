#!/usr/bin/env python3

### GET z_dim FROM APPROX LD BLOCKS ###

import numpy as np
import os
import pandas as pd
from scipy.linalg import schur
from tqdm import tqdm
import argparse

from utils import import_data, load_gene_matrix

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Find z dimension by LD blocks approximation.')

# Add arguments
parser.add_argument('gene_file', type=str, help='Name of gene list file.')

# Parse the arguments
args = parser.parse_args()
gene_list_name = args.gene_file

# Change working directory
# os.chdir('/media/HDD_4TB_1/jordi/cfuses_gnn_enrollhd_2024/')
os.chdir('/gpfs/projects/ub112')

# Input and output directories
# data_dir = "data/features/"
# biomart_dir = "data/biomart/"
# output_dir = "data/vae/"
data_dir = "data/enroll_hd/features/"
biomart_dir = "data/enroll_hd/biomart/"

# Data path
# X_path = data_dir + "X_pc10_filt_0.01.txt"
# header_path = data_dir + "subsetting/header_X_pc10_filt_0.01.txt"

X_path = data_dir + "feature_matrix_m3_filt_0.01.txt"
header_path = data_dir + "header_feature_matrix_m3_filt_0.01.txt"

lookuptab_path = biomart_dir + "revised_filtered_snp_gene_lookup_tab.txt"

# gene_list_path = "data/genes/revised_core_genes_names.txt"
# gene_list_path = data_dir + "revised_core_genes_names.txt"
gene_list_path = data_dir + gene_list_name

output_path = data_dir + 'zdims_' + gene_list_name

# Obtain gene_matrix
X, lookuptab, header = import_data(X_path=X_path, 
                                    lookuptab_path=lookuptab_path,
                                    header_path=header_path)

# Get genes list
gene_list = pd.read_csv(gene_list_path)

# Create output lists
genes = []
z_dims = []
n_snps = []

print("Data loaded")

# Iterate over genes
for gene in tqdm(gene_list['Gene'], desc="Processing"):
    # Subset feature matrix
    gene_matrix = load_gene_matrix(X, lookuptab, header, gene = gene)
    
    # Ensure the gene has SNPs
    if gene_matrix.shape[1] == 0:
        continue
    
    # Create a DataFrame
    snps_df = pd.DataFrame(gene_matrix)

    # Compute the correlation matrix
    snps_corr_matrix = snps_df.corr()
    
    # Compute the Schur decomposition
    T, _ = schur(abs(snps_corr_matrix))

    # Extract the eigenvalues from the diagonal of T
    eigenvalues = np.diag(T)

    # Find how many eigenvalues to take as z_dim
    norm_eigens = eigenvalues/max(eigenvalues)

    # Minimum accepted normalized eigenvalue
    threshold = 0.2
    z_dim = 0

    for eigen in norm_eigens:
        if eigen >= threshold:
            z_dim += 1
    
    # Append results
    genes.append(gene)
    z_dims.append(z_dim)
    n_snps.append(gene_matrix.shape[1])

    #print(gene,z_dim)
    
# Create output file
results = pd.DataFrame({'Gene':genes, 'z_dim':z_dims, 'n_snps':n_snps})
print("Number of genes:", len(results))
results.to_csv(output_path, index=False, sep='\t', header=None)