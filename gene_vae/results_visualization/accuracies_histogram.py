#!/usr/bin/env python3

### TRAIN MODELS WITH 100 DIFFERENT DATA SPLITS ###
#%%
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

os.chdir('/pool01/projects/abante_lab/ao_prediction_enrollhd_2024/')
data_dir  = "vae/"

out_dir = "ml_results/classification/v2/paperfigs/"

# Import data
metrics_df = pd.read_csv(data_dir + "reconstruction_metrics_allgenes.tsv", sep='\t')

dim_reduction_df = pd.read_csv('features/zdims_genes.txt', sep='\t', header=None)
dim_reduction_df = dim_reduction_df.rename({0:'gene', 1:'zdim', 2:'n_snps'}, axis=1)

# %%
# Stacked histograms 

# Create figure and subplots
fig, axs = plt.subplots(3, 1, figsize=(6, 10), sharex=True)

# Plot histograms for each column in a stacked manner
axs[0].hist(metrics_df['zero_acc'], bins=40, color='grey')
axs[0].set_title('Zero Accuracy')

axs[1].hist(metrics_df['one_acc'], bins=40, color='grey')
axs[1].set_title('One Accuracy')

axs[2].hist(metrics_df['two_acc'], bins=40, color='grey')
axs[2].set_title('Two Accuracy')

# Label x-axis and adjust layout
plt.xlabel('Accuracy')
plt.tight_layout()

# Show plot
plt.show()
# %%
# Overlapped histograms

# Create figure and axis
plt.figure(figsize=(8, 6))

# Plot histograms for each column on the same axis with transparency
plt.hist(metrics_df['zero_acc'], bins=40, color='blue', alpha=0.5, label='Zero Accuracy')
plt.hist(metrics_df['one_acc'], bins=40, color='green', alpha=0.5, label='One Accuracy')
plt.hist(metrics_df['two_acc'], bins=40, color='red', alpha=0.5, label='Two Accuracy')

# Add title, labels, and legend
plt.title('Layered Accuracy Histograms')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.legend(loc='upper left')

# Show plot
plt.show()

# %%
# Merge two dataframes
results_df = pd.merge(metrics_df, dim_reduction_df, how = 'inner', on='gene')

# %%
# Dimensionality reduction column
results_df['dim_reduction'] = results_df['zdim_y']/results_df['n_snps']
# %%
# Take all genes where 1s accuracy is 0
zero_acc_1s = results_df[results_df['one_acc']==0]
zero_acc_2s = results_df[results_df['two_acc']==0]

# %%
fig, axs = plt.subplots(1,3, figsize=(12,6))
# Plot histograms
axs[0].hist(zero_acc_1s['zdim_y'], bins=20, color='blue', alpha=0.7)
axs[0].set_ylabel('frequency')
axs[0].set_title('z dimension')

axs[1].hist(zero_acc_1s['n_snps'], bins=20, color='green', alpha=0.7)
axs[1].set_title('# snps')

axs[2].hist(zero_acc_1s['dim_reduction']*100, bins=20, color='red', alpha=0.7)
axs[2].set_title('dimensionality reduction (%)')

fig.suptitle('Genes with 1s accuracy = 0')

plt.show()

# %%
fig, axs = plt.subplots(1,3, figsize=(12,6))
# Plot histograms
axs[0].hist(zero_acc_2s['zdim_y'], bins=20, color='blue', alpha=0.7)
axs[0].set_ylabel('frequency')
axs[0].set_title('z dimension')

axs[1].hist(zero_acc_2s['n_snps'], bins=20, color='green', alpha=0.7)
axs[1].set_title('# snps')

axs[2].hist(zero_acc_2s['dim_reduction']*100, bins=20, color='red', alpha=0.7)
axs[2].set_title('dimensionality reduction (%)')

fig.suptitle('Genes with 2s accuracy = 0')

plt.show()
# %%
# Extract observations with highest dimensionality reduction

zero_acc_1s_high_red = zero_acc_1s[zero_acc_1s['dim_reduction']>0.9]

fig, axs = plt.subplots(1,2, figsize=(8,6))
# Plot histograms
axs[0].hist(zero_acc_1s_high_red['zdim_y'], bins=20, color='blue', alpha=0.7)
axs[0].set_ylabel('frequency')
axs[0].set_title('z dimension')

axs[1].hist(zero_acc_1s_high_red['n_snps'], bins=20, color='green', alpha=0.7)
axs[1].set_title('# snps')

fig.suptitle('Genes with 1s accuracy = 0 and a dimensionality reduction > 90%')

plt.show()
# %%

metrics_dim_red = results_df[results_df['n_snps']>30]

# Stacked histograms 

# Create figure and subplots
fig, axs = plt.subplots(3, 1, figsize=(6, 7), sharex=True)

# Plot histograms for each column in a stacked manner
axs[0].hist(metrics_dim_red['zero_acc'], bins=3, color='grey')
axs[0].set_title('Zero Accuracy')

axs[1].hist(metrics_dim_red['one_acc'], bins=40, color='grey')
axs[1].set_title('One Accuracy')

axs[2].hist(metrics_dim_red['two_acc'], bins=40, color='grey')
axs[2].set_title('Two Accuracy')

# Label x-axis and adjust layout
plt.xlabel('Accuracy')
fig.suptitle('Genes with #snps > 30 (N = {})'.format(metrics_dim_red.shape[0]))
plt.tight_layout()

plt.savefig(f'{out_dir}vae_accuracies_hist.pdf', dpi=300)

# Show plot
plt.show()
# %% Mean accuracy
metrics_df['Mean_acc'] = np.mean([metrics_df['zero_acc'], metrics_df['one_acc'], metrics_df['two_acc']], axis = 0)

# %%
