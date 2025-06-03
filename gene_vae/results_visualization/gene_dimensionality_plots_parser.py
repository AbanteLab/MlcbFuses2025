#!/usr/bin/env python3

### PLOT GENE SNPS CORRELATION AND Z DIM ###

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse
from scipy.linalg import schur
from sklearn.preprocessing import normalize
import pyemma.msm as msm

from utils import import_data, load_gene_matrix

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Gene to represent')

# Add arguments
parser.add_argument('gene_name', type=str, help='Name of gene to represent')

# Parse the arguments
args = parser.parse_args()
gene = args.gene_name

os.chdir('/home/ub/ub781464/')

data_dir = "data/enroll_hd/features/"
biomart_dir = "data/enroll_hd/biomart/"
output_dir = "data/enroll_hd/ld_blocks/"

X_path = data_dir + "feature_matrix_m3_filt_0.01.txt"
header_path = data_dir + "subsetting/header_feature_matrix_m3_filt_0.01.txt"

lookuptab_path = biomart_dir + "revised_filtered_snp_gene_lookup_tab.txt"

# Obtain gene_matrix
X, lookuptab, header = import_data(X_path=X_path, 
                                    lookuptab_path=lookuptab_path,
                                    header_path=header_path)

# Subset feature matrix
gene_matrix = load_gene_matrix(X, lookuptab, header, gene = gene)

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

# Normalize correlation absolute values to obtain transition probabilities
normed_matrix = normalize(abs(snps_corr_matrix), axis=1, norm='l1')

# Create a Markov state model (MSM)
msm_model = msm.markov_model(normed_matrix)

# Perform GPCCA to identify metastable states
n_metastable_states = 27
gpcca = msm_model.pcca(n_metastable_states)

# Coarse-grained transition matrix
P_coarse = gpcca.coarse_grained_transition_matrix

# Memberships of each state in the metastable sets
memberships = gpcca.memberships

# Plot

fig, axes = plt.subplots(2,2, figsize=(12, 10))

fig.suptitle(gene, fontsize=16)

# Unpack axes from the 2x2 array
ax11, ax12 = axes[0]
ax21, ax22 = axes[1]

# Correlation matrix
sns.heatmap(abs(snps_corr_matrix), cmap='OrRd', vmin=0, vmax=1, ax=ax11)
ax11.set_title('SNPs Absolute Correlation Coefficient Matrix')

# Plot eigenvalues
ax21.plot(eigenvalues[:30])
ax21.set_xlabel("Eigenvalue number")
ax21.set_ylabel("Eigenvalue")
ax21.set_title("SNPs Correlation Matrix Eigenvalues")

# Plot heatmap of the coarse-grained transition matrix (macrostates)
sns.heatmap(P_coarse, cmap='viridis', cbar=True, ax=ax12)
ax12.set_title('Coarse-grained Transition Matrix (Macrostates)')
ax12.set_xlabel('Macrostate')
ax12.set_ylabel('Macrostate')

# Plot heatmap of the memberships
sns.heatmap(memberships, cmap='viridis', cbar=True, ax = ax22)
ax22.set_title('Memberships in Metastable States')
ax22.set_xlabel('Metastable State')
ax22.set_ylabel('Original State')

plt.tight_layout(rect=[0, 0, 1, 0.98]) 

plt.savefig(output_dir + f'{gene}_ld.png')