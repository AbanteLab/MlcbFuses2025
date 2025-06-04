#!/usr/bin/env python3

import os
import time
import numpy as np
from tqdm import tqdm
import argparse

from datetime import datetime
def _print(*args, **kw):
    print("[%s]" % (datetime.now()),*args, **kw)
    
# Feature matrix characteristics (m3)
# Real: n_samples = 9064 / n_snps = 613467
# Test: n_samples = 900 / n_snps = 70000

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Filters (patient,SNP) matrix for a minimum prevalence.')

# Add arguments
parser.add_argument('matrix', type=str, help='Input (patient,SNP) matrix')
parser.add_argument('n_samples', type=int, help='Number of samples in input matrix')
parser.add_argument('n_snps', type=int, help='Number of SNPs in input matrix')
parser.add_argument('thres', type=float, help='Minimum prevalence threshold')

# Parse the arguments
args = parser.parse_args()

# Extract the prefix from the matrix filename
prefix = os.path.splitext(os.path.basename(args.matrix))[0]

# Construct the output filename
output_filename = f"{prefix}_filt_{args.thres}.txt"

# define minimum prevalence in terms of patients
min_alt_prev_threshold = args.n_samples * args.thres

# Initialize the vector storing number of samples with alternative allele for every SNP
vector = np.zeros(args.n_snps, dtype=np.uint8)

_print("Checking minimum prevalence...")

# Open the text file
with open(args.matrix, 'r') as file:
    
    # Skip the first line
    next(file)
        
    # Iterate over each line in the file for line in file:
    for line_number, line in tqdm(enumerate(file, start=1), total=args.n_samples):
        
        # Split the line by tab
        snps = line.strip().split("\t")
        
        # Iterate over snps, start from 4th position to skip id, sex, cag
        for i in range(3, len(snps)):
            
            # Convert the element to an integer (uint8)
            value = np.uint8(snps[i])
            
            # If the element is different from 0, increment the corresponding position in the vector
            if value != 0:
                vector[i - 3] += 1

# Get indices surpassing minimum                
mask = vector >= min_alt_prev_threshold
mask = np.concatenate((np.array([True,True,True]), mask))

_print("Storing filtered matrix...")

# Open the text file
with open(output_filename, 'w') as wr_file:
    with open(args.matrix, 'r') as rd_file:
            
        # Iterate over each line in the file
        for line_number, line in tqdm(enumerate(rd_file, start=1), total=args.n_samples):
            
            # Split the line by tab
            snps = np.array(line.strip().split("\t"))[mask]
            
            # write
            wr_file.write('\t'.join(map(str, snps)))
            wr_file.write('\n')
        