#!/usr/bin/env python3

### Filters the feature matrix by the prevalence of the alternative variant ###

import os
import pandas as pd
import numpy as np
import pickle
import time

# Change working directory
os.chdir('/media/HDD_4TB_1/jordi/cfuses_gnn_enrollhd_2024/')

# Input and output directories
data_dir = "data/features/"

# Record the start time
start_time = time.time()

# Read the first line of the txt file to get column names
with open(data_dir + "feature_matrix_m3.txt") as f:
    first_line = f.readline().strip()
column_names = first_line.split("\t")[1:]

# Create a dictionary to specify data types for all columns except the first one
dtype_dict = {col: 'uint8' for col in column_names}

# Read the CSV file with specified data types
X = pd.read_csv(data_dir + "feature_matrix_m3.txt", sep = "\t", dtype=dtype_dict, engine='c')

# Record the end time
end_time = time.time()

# Compute the elapsed time
elapsed_time = end_time - start_time

print("Time taken to load the table: {:.2f} seconds".format(elapsed_time))

# Set sample identificator as table index
X.set_index('FID_IID', inplace=True)

# Get rid of sex and CAG columns
X.drop('Sex', axis = 1, inplace=True)
X.drop('CAG', axis = 1, inplace=True)

# Total number of samples
n_samples = len(X)

# SNP names in column order
snp_names = X.columns

# Initialize vector to store prevalence of alternative variant, always taking first two
rowsums = []

# Sum all rows of each column
for col in snp_names:
    col_sum = sum([1 for val in X[col] if val != 0])
    rowsums.append(col_sum)
    
print('Iterative sum across rows completed.')

def min_prevalence_filtering(min_prevalence, rowsums, snp_names, n_samples): 
    '''
    Returns columns of matrix X (SNPs) whose proportion of alternative allele across
    all samples is higher than the minimum specified as parameter.
    
    INPUT:
    min_prevalence: minimum prevalence of alternative variant to set
    rowsums: list of sum across all samples for each SNP (generated +1 if genotype != 0) 
    snp_names: list with SNP names following column order as feature matrix
    n_samples: number of samples in feature matrix
    
    OUTPUT:
    n_approved_snps = number of SNPs whose sum >= minimum set
    approved_snps = name of "
    '''       
        
    # Variable to count how many SNPs pass the threshold 
    n_approved_snps = 0

    # List to save approved snp names
    approved_snps = []

    # Check each SNP sum across samples
    for idx,samplesum in enumerate(rowsums):
        if samplesum >= min_prevalence*n_samples:
            
            # Increase counter
            n_approved_snps +=1
            
            # Add snp name
            approved_snps.append(snp_names[idx])
            
    return n_approved_snps, approved_snps
    
# Values of minimum prevalence to test
minvals = np.arange(0,1.1,0.1)

# SNPs filtered in each val
snps_filtered_per_val = {}

# Save n_approved_snps for each minimum prevalence value
dependant_approved_snps = []
for val in minvals:
    n_approved_snps, approved_snps = min_prevalence_filtering(val, rowsums, snp_names, n_samples)
    dependant_approved_snps.append(n_approved_snps)
    snps_filtered_per_val[val] = approved_snps

print('Saving results...')

# Pickle dictionary
with open(data_dir + 'subsetting/' + 'snps_filtered_per_val.pkl', 'wb') as f:
    pickle.dump(snps_filtered_per_val, f)
    
# Pickle rowsums for further testing
with open(data_dir + 'subsetting/' + 'gen_different_from_0_sum.pkl', 'wb') as f:
    pickle.dump(rowsums, f)

# Create dataframe with results for display
#prevalence_filtering_sizes = pd.DataFrame({'Minimum alternative prevalence':minvals*100, 'Number of SNPs':dependant_approved_snps})

# Pickle dataframe
#with open(data_dir + 'subsetting/' + 'prevalence_filtering_sizes.pkl', 'wb') as f:
    #pickle.dump(prevalence_filtering_sizes, f)