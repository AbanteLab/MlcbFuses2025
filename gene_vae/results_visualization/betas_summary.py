#!/usr/bin/env python3

### PLOT THE RESULTS OF VAEs WITH DIFFERENT HYPERPARAMETERS ###

#%%
import itertools
import numpy as np
import subprocess
import os
import re
import matplotlib.pyplot as plt

os.chdir('/media/HDD_4TB_1/jordi/cfuses_gnn_enrollhd_2024/')

#%%

gene = 'WWOX'    
#%% Read objective functions for each combination ###

# Directory containing .npz files
directory = "data/vae/"

# Initialize a dictionary to store the results
results = {}

# Regular expression to extract values from filenames
filename_pattern = re.compile(
    r'objective_functions_{}_fdecBernoulli_zdim(\d+)_hdimprop(\d+)_beta(\d+\.\d+).npz'.format(gene))

# Iterate over each .npz file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.npz'):
        match = filename_pattern.match(filename)
        if match:
            zdim = int(match.group(1))
            hdimprop = int(match.group(2))
            beta = float(match.group(3))
            
            # Load the .npz file
            file_path = os.path.join(directory, filename)
            with np.load(file_path) as data:
                test_losses = data['test_elbo'].tolist()  
                reconstruction_losses = data['test_rl'].tolist()  
                kl_divergences = data['test_kl'].tolist()  
                total_losses = data['test_custom_elbo'].tolist()  
            
            # Organize data in a nested dictionary
            if (zdim, hdimprop) not in results:
                results[(zdim, hdimprop)] = {'beta':[],
                                             'test_elbo':[],
                                             'reconstruction_loss':[],
                                             'kl_divergence':[],
                                             'total_loss':[]}
                
            results[(zdim, hdimprop)]['beta'].append(beta)
            results[(zdim, hdimprop)]['test_elbo'].append(test_losses[-1])
            results[(zdim, hdimprop)]['reconstruction_loss'].append(reconstruction_losses[-1])
            results[(zdim, hdimprop)]['kl_divergence'].append(kl_divergences[-1])
            results[(zdim, hdimprop)]['total_loss'].append(total_losses[-1])

#%% Plot them

fig, axs = plt.subplots(2, figsize=(5, 8))
fig.suptitle('Losses of {} training'.format(gene))

# Dictionary to hold labels and their corresponding lines
lines = {}
labels = set()

# Plot the data
for zhcombination, data in results.items():
    label = "zdim {}, hdimprop {}".format(zhcombination[0], zhcombination[1])
    
    # Sort the data by beta
    sorted_indices = np.argsort(data['beta'])
    sorted_beta = np.array(data['beta'])[sorted_indices]
    sorted_reconstruction_loss = np.array(data['reconstruction_loss'])[sorted_indices]
    sorted_kl_divergence = np.array(data['kl_divergence'])[sorted_indices]
    
    # Plot reconstruction loss with dots
    line, = axs[0].plot(sorted_beta, sorted_reconstruction_loss, label=label, marker='o', linestyle='-', markersize=5)
    labels.add(label)
    lines[label] = line
    
    # Plot KL divergence with dots
    axs[1].plot(sorted_beta, sorted_kl_divergence, marker='o', linestyle='-', markersize=5, color=line.get_color())

# Add legend
fig.legend(handles=[lines[label] for label in labels], loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2)

# Add labels to the axes
axs[0].set_ylabel('Reconstruction Loss')
axs[1].set_xlabel('Beta')
axs[1].set_ylabel('KL Divergence')

# Adjust layout to make room for the legend
plt.subplots_adjust(top=0.9)
plt.tight_layout(rect=[0, 0, 1, 0.90])

plt.show()
