#!/usr/bin/env python3

### PLOT THE RESULTS OF VAEs WITH DIFFERENT HYPERPARAMETERS ###

import numpy as np
import os
import re
import matplotlib.pyplot as plt
import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Plot vae losses for different hidden layer dimensions.')

# Add arguments
parser.add_argument('gene', type=str, help='Gene to subset from feature matrix.')
parser.add_argument('z_dim', type=int, help='Dimensionality of latent space.')

# Parse the arguments
args = parser.parse_args()
gene = args.gene
zdim = args.z_dim

# Directories

os.chdir('/gpfs/projects/ub112')
# Directory containing .npz files
directory = "data/enroll_hd/vae/"

# Initialize a dictionary to store the results
results = {}

# Regular expression to extract values from filenames
filename_pattern = re.compile(
    r'objective_functions_{}_fdecBernoulli_zdim{}_hdimprop(\d+)_beta1.0.npz'.format(gene, zdim))

# Iterate over each .npz file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.npz'):
        match = filename_pattern.match(filename)
        if match:
            hdimprop = int(match.group(1))
            
            # Load the .npz file
            file_path = os.path.join(directory, filename)
            with np.load(file_path) as data:
                test_losses = data['test_elbo'].tolist()  
                reconstruction_losses = data['test_rl'].tolist()  
                kl_divergences = data['test_kl'].tolist()  
                total_losses = data['test_custom_elbo'].tolist()  
            
            # Organize data in a nested dictionary
            if hdimprop not in results:
                results[hdimprop] = {'test_elbo':[],
                                    'reconstruction_loss':[],
                                    'kl_divergence':[],
                                    'total_loss':[]}

            results[hdimprop]['test_elbo'].append(test_losses[-1])
            results[hdimprop]['reconstruction_loss'].append(reconstruction_losses[-1])
            results[hdimprop]['kl_divergence'].append(kl_divergences[-1])
            results[hdimprop]['total_loss'].append(total_losses[-1])

# Sort the hdimprop values for consistent plotting
sorted_hdimprop = sorted(results.keys())

# Prepare the data for plotting
reconstruction_losses = [results[hdim]['reconstruction_loss'] for hdim in sorted_hdimprop]
kl_divergences = [results[hdim]['kl_divergence'] for hdim in sorted_hdimprop]

#%% Plot them

fig, axs = plt.subplots(2, figsize=(5, 8))
fig.suptitle('Losses of {} training ({} z dimensions)'.format(gene, zdim))

axs[0].plot(sorted_hdimprop, reconstruction_losses, marker='o', linestyle='-', color='b')
axs[1].plot(sorted_hdimprop, kl_divergences, marker='o', linestyle='-', color='b')

# Add labels to the axes
axs[0].set_ylabel('Reconstruction Loss')
axs[1].set_xlabel('Hidden layer dimension')
axs[1].set_ylabel('KL Divergence')

plt.tight_layout()

plt.savefig(directory + "summary_hidden_dims_{}.png".format(gene))