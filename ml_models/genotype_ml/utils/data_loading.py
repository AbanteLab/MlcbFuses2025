#!/usr/bin/env python3

### Functions for data loading ###

# Dependencies
from scipy.sparse import csr_matrix, vstack
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import os
import torch
from scipy.sparse import load_npz

def _print(*args, **kw):
    # Printing time for log recording
    print("[%s]" % (datetime.now()),*args, **kw)

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

def scale_CAG(X):
    '''Scales CAG column (second column of X) to have range 0-1.'''
    # Extract the second column of the CSR matrix
    CAG_column = X[:, 1]

    # Convert the extracted column to a dense numpy array
    dense_CAG = CAG_column.toarray().reshape(-1, 1)

    # Find the min and max values of CAG column
    min_cag = dense_CAG.min()
    max_cag = dense_CAG.max()

    # Scale CAG column to range 0-1
    scaled_CAG = (dense_CAG - min_cag) / (max_cag - min_cag)

    # Replace CAG column by scaled values
    X[:, 1] = csr_matrix(scaled_CAG)
    
    return X

def interaction_computation(X, chunk_size=100):
    '''Multiplies all SNP genotypes (0, 1, or 2) in matrix X
    by the scaled CAG value of the corresponding subject.'''
    
    num_rows, num_cols = X.shape
    sparse_chunks = []
    
    for i in range(0, num_cols, chunk_size):
        # Get the chunk of rows
        dense_chunk = X[i:min(i+chunk_size, num_cols), :].toarray()
        
        # Multiply all columns starting from the third one by 1 - CAG value
        for j in range(2, num_cols):
            dense_chunk[:, j] *= dense_chunk[:, 1]
        
        # Sparsify back the chunk
        sparse_chunk = csr_matrix(dense_chunk)
        
        # Append the chunk to the list
        sparse_chunks.append(sparse_chunk)
    
    # Concatenate the list of CSR matrices into a single CSR matrix
    result = vstack(sparse_chunks)
    
    return result

def load_X_y(X_path, y_path):
    """
    Load feature matrix X and labels y.
    - If X_path is a .txt file, it loads X as a sparse matrix.
    - If X_path is a .pth file, it loads X as a PyTorch tensor and converts it to a CSR matrix.
    - If X_path is a .npz file, it loads X as a numpy matrix and DOES NOT scale CAG column.
    """
    if X_path.endswith(".txt"):
        # Load X
        X, _ = read_sparse_X(X_path, chunk_size=100)

        # Scale CAG column
        X = scale_CAG(X)

    elif X_path.endswith(".pth"):
        X = torch.load(X_path).numpy()
        
        # Load sexCAG.npy and prepend as columns
        sexCAG_path = os.path.join(os.path.dirname(X_path), "sexCAG.npy")
        if os.path.exists(sexCAG_path):
            sexCAG = np.load(sexCAG_path)
            X = np.hstack((sexCAG, X))
        else:
            raise FileNotFoundError(f"Missing required file: {sexCAG_path}")
        
        # Convert to csr matrix
        X = csr_matrix(X)

        # Scale CAG column
        X = scale_CAG(X)

    elif X_path.endswith(".npz"):
        X = load_npz(X_path)

    else:
        raise ValueError("Unsupported file format. Use .txt, .pth or .npz")

    # Load outcome vector
    y = pd.read_csv(y_path, header=None).squeeze("columns").to_numpy()

    _print("Data loaded.")
    
    return X, y