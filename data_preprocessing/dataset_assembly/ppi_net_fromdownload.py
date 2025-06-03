#!/usr/bin/env python3

### PPI network generation ###
#%%
import pandas as pd
import numpy as np
import os
import pickle
from scipy.sparse import lil_matrix
from datetime import datetime
import time
from tqdm import tqdm

def _print(*args, **kw):
    print("[%s]" % (datetime.now()), *args, **kw)

_print('Starting time.')

os.chdir('/media/HDD_4TB_1/creatio/ao_prediction_enrollhd_2024/')

outdir = "data/ppi_networks/"

gene_list_file = "data/features/zdims_genes.txt"
alias_file = outdir + "9606.protein.aliases.v12.0.txt.gz"
string_file = outdir + "9606.protein.links.detailed.v12.0.txt.gz"

# Load the list of our genes
genes_df = pd.read_csv(gene_list_file, sep='\t', header=None)
genes = set(genes_df[0].tolist())  # Set of unique genes for quick lookup

# Load the STRING alias file to map gene symbols to ENSP identifiers
aliases = pd.read_csv(alias_file, sep="\t", header=None, names=["protein", "alias", "source"], compression="gzip")

# Take aliases from a single source
aliases = aliases[aliases['source']=='Ensembl_UniProt']

# Filter aliases to include only those mapping our gene symbols to ENSP identifiers
symbol_to_ensp = aliases[(aliases["alias"].isin(genes)) & (aliases["protein"].str.startswith("9606.ENSP"))]
symbol_to_ensp = symbol_to_ensp[["alias", "protein"]].drop_duplicates()

# Deduplicate to keep only one ENSP per gene symbol (if multiple protein isoforms exist)
unique_symbol_to_ensp = symbol_to_ensp.drop_duplicates(subset="alias", keep="first")

# Create a DataFrame with Symbol, ENSP, and unique Index
symbol_to_ensp_df = unique_symbol_to_ensp.reset_index(drop=True).copy()
symbol_to_ensp_df["Index"] = symbol_to_ensp_df.index  # Assign a unique index to each ENSP

# Convert DataFrame to dictionary for mapping symbols to ENSP identifiers
symbol_to_ensp_dict = dict(zip(symbol_to_ensp_df["alias"], symbol_to_ensp_df["protein"]))
genes_ensp = set(symbol_to_ensp_df["protein"].tolist())

_print(f"Mapped {len(symbol_to_ensp_dict)} unique gene symbols to {len(genes_ensp)} unique ENSP identifiers.")

unmapped_genes = genes - set(symbol_to_ensp_dict.keys())
_print(f"{len(unmapped_genes)} gene symbols could not be mapped to an ENSP identifier.")
_print("Unmapped gene symbols:", unmapped_genes)

# Load the STRING interactions data
interactions = pd.read_csv(string_file, sep=" ", usecols=["protein1", "protein2", "experimental"], compression="gzip")

# Filter interactions for ENSP identifiers in our gene list
filtered_interactions = interactions[
    interactions["protein1"].isin(genes_ensp) & interactions["protein2"].isin(genes_ensp)
]

# Map ENSP identifiers in filtered_interactions to indices using symbol_to_ensp_df
ensp_to_index = dict(zip(symbol_to_ensp_df["protein"], symbol_to_ensp_df["Index"]))
protein1_indices = filtered_interactions["protein1"].map(ensp_to_index).values
protein2_indices = filtered_interactions["protein2"].map(ensp_to_index).values
scores = filtered_interactions["experimental"].values

# Create an adjacency matrix using the mapped indices
n = len(symbol_to_ensp_df)
adjacency_matrix = lil_matrix((n, n))

# Populate the adjacency matrix using numpy
for i, j, score in tqdm(zip(protein1_indices, protein2_indices, scores), total=filtered_interactions.shape[0], desc="Building adjacency matrix"):
    adjacency_matrix[i, j] = score
    adjacency_matrix[j, i] = score  # Ensure symmetry
#%%
# Save the adjacency matrix
with open(outdir + "protein_interactions_matrix_filtered.pkl", "wb") as f:
    pickle.dump(adjacency_matrix, f)

# Save the mapping DataFrame as a CSV with Symbol, ENSP ID, and Index
symbol_to_ensp_df.to_csv(outdir + "protein_interactions_matrix_index_mapping.csv", index=False)

_print("Filtered adjacency matrix and labels saved.")