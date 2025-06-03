#!/usr/bin/env python3

### PLOT THE LATENT VARIABLES DISTRIBUTION ###

#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap
import os
import torch

from utils import import_data, load_gene_matrix, encode_data
from vae_classes import VAE

#%% Data loading

os.chdir('/media/HDD_4TB_1/jordi/cfuses_gnn_enrollhd_2024/')
data_dir = "data/features/"
biomart_dir = "data/biomart/"
output_dir = "data/vae/"

X_path = data_dir + "feature_matrix_m3_filt_0.01.txt"
header_path = data_dir + "subsetting/header_feature_matrix_m3_filt_0.01.txt"

lookuptab_path = biomart_dir + "revised_filtered_snp_gene_lookup_tab.txt"

#%%
gene = "WWOX"
z_dim = 10
hdim_prop = 40
beta = 1.5
f_dec = "Bernoulli"

model_path = output_dir + "model_{}_fdec{}_zdim{}_hdimprop{}_beta{}".format(gene, f_dec, z_dim, hdim_prop, beta)

#%% Load feature matrix
X, lookuptab, header = import_data(X_path=X_path, 
                                    lookuptab_path=lookuptab_path,
                                    header_path=header_path)
    
#%% Subset feature matrix
gene_matrix = load_gene_matrix(X, lookuptab, header, gene = gene)

#%%

if f_dec == "Bernoulli":
    # Normalize data between 0 and 1
    gene_matrix = gene_matrix*0.5
    
# Variable that stores input size (number of SNPs in gene)
g_length = gene_matrix.shape[1]

#%% Calculate dimensionality of hidden layer
hidden_dim = int(round(g_length * (hdim_prop/100), 0))

# Load trained vae model
vae = VAE(z_dim = z_dim, hidden_dim=hidden_dim, gene_length=g_length, f_dec=f_dec, beta=beta)
vae.load_state_dict(torch.load(model_path))

#%% Encode
encoded_data = encode_data(vae, gene_matrix)

#%% UMAP fit
reducer = umap.UMAP()
embedding = reducer.fit_transform(encoded_data)
embedding.shape

#%% Plot
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    s=10,   
    alpha=0.5)
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of {} latent space (dim {})'.format(gene, z_dim, beta), fontsize=15)

output_sufix = "_{}_fdec{}_zdim{}_hdimprop{}_beta{}".format(gene, f_dec, z_dim, hdim_prop, beta)

plt.savefig(output_dir + "z_umap" + output_sufix + ".png")