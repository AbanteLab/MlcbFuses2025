#!/usr/bin/env python3

#%%## weight logSED ###

import pandas as pd
import numpy as np
import polars as pl
import re
import ast
import os 
from scipy.stats import norm
import argparse

#%%
parser = argparse.ArgumentParser(description="Weight Borzoi logSED values.")
parser.add_argument("--chromosome", required=True, type=str, help="chromosome number to process (ex: 18)")
args = parser.parse_args()
chromosome = args.chromosome
#%%
# Set working directory
os.chdir("/pool01/projects/abante_lab")

out_dir = "genomic_llms/borzoi/proc_results/weighted_logSED/"

vcf_path = "ao_prediction_enrollhd_2024/enroll_hd/regulatory_vcfs/"
vcf_file = f"{vcf_path}gwa12345.mis4.9064.hg38.cisreg0.5.mad0.01.chr{chromosome}.vcf"

enhancer_path = "ao_prediction_enrollhd_2024/genes/gene_cisreg.txt"
predictions_path = f"genomic_llms/borzoi/proc_results/sed_all_chr/sed_chr{chromosome}.txt.gz"

# Load Enhancer File 
enh = pd.read_csv(enhancer_path, sep="\t")  # uses header from file

# Convert positions to integers
enh["start"] = enh["start"].astype(int)
enh["end"] = enh["end"].astype(int)

# Normalize chromosome naming (remove 'chr')
enh["chrom"] = enh["chrom"].str.replace("chr", "", regex=False)

# Filter to target chromosome only 
enh = enh[enh["chrom"] == chromosome]

# Convert connected_genes from string to dictionary
enh['connected_genes'] = enh['connected_genes'].apply(
    lambda x: ast.literal_eval(x) if pd.notna(x) else x) # Skip promoters

# Extract connected genes keys (discard scores)
enh["gene_all_ids"] = enh["connected_genes"].apply(lambda d: list(d.keys()))

#%%
# Keep only specified columns
enh = enh[["feature name", "genehancer_id", "gene_all_ids", "chrom", "start", "end"]]

# Read the VCF file while skipping headers (lines starting with '##')
with open(vcf_file, "r") as f:
    header_lines = [line for line in f if line.startswith("#")]

# Load VCF data into a Polars DataFrame (skipping header lines)
vcf_df = pl.read_csv(vcf_file, comment_prefix="#", separator="\t", has_header=False)

# Assign column names (VCF standard format)
vcf_df = vcf_df.rename({
    "column_1": "CHROM",
    "column_2": "POS",
    "column_3": "ID",
    "column_4": "REF",
    "column_5": "ALT",
    "column_6": "QUAL",
    "column_7": "FILTER",
    "column_8": "INFO",
    "column_9": "FORMAT"
})
vcf_df = vcf_df.with_columns(pl.col("POS").cast(pl.Int64))

# Filter vcf_df for the current chromosome
vcf_df = vcf_df.filter(pl.col("CHROM") == 'chr' + str(chromosome))

# Load Predictions File 
preds = pl.read_csv(predictions_path, separator="\t")

# Join on 'snp' == 'ID'
preds = preds.join(vcf_df.select(['CHROM', 'POS', 'ID']), left_on='snp', right_on='ID', how='left')

# Extract numeric chrom and pos
preds = preds.with_columns([
    pl.col('CHROM').str.replace('chr', '').cast(pl.Int64).alias('chrom'),
    pl.col('POS').cast(pl.Int64).alias('pos')
])
preds = preds.drop(['CHROM', 'POS'])

# Create a list of tuples with start, end, genehancer_id
intervals = list(zip(enh['start'].to_list(), enh['end'].to_list(), enh['genehancer_id'].to_list()))

# Define a function that returns all genehancer_ids that match the position
def match_enhancer(pos):
    matches = [genehancer_id for start, end, genehancer_id in intervals if start <= pos <= end]
    if matches:
        return matches[0]  # always first match as string
    return None

# Apply this function to each row's pos in preds
preds = preds.with_columns([
    pl.col('pos').apply(match_enhancer).alias('genehancer_id')
])

sigma = 300  # Gaussian std dev from the paper

def compute_weighted_logsed(row):
    enhancer_id = row["genehancer_id"]
    pos = row["pos"]
    logsed = row["logSED"]
    
    # Skip weighting for promoters or null values
    if enhancer_id is None or enhancer_id.startswith("Promoter_"):
        return logsed

    # Get enhancer row from Pandas DataFrame
    gene_enhancers = enh[enh["genehancer_id"] == enhancer_id]
    if gene_enhancers.empty:
        return 0.0

    # Compute enhancer center and Gaussian weight
    enh_row = gene_enhancers.iloc[0]  # taking first match
    center = (enh_row["start"] + enh_row["end"]) / 2
    distance = center - pos
    weight = norm.pdf(distance, loc=0, scale=sigma)

    return logsed * weight

# Apply row-wise logic
preds = preds.with_columns([
    pl.struct(["genehancer_id", "pos", "logSED"])
      .map_elements(compute_weighted_logsed)
      .alias("weighted_logSED")
])

preds.write_csv(f"{out_dir}chr{chromosome}_weighted_predictions.tsv", separator="\t")
# %%
print(f"Results saved in chr{chromosome}_weighted_predictions.tsv")