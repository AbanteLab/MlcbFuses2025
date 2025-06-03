#!/usr/bin/env python3

### Assembles vcf files with SNPs from regulatory regions of our genes ###
#%%
# Dependencies (conda environment: enrollhd)
import os 
import pandas as pd
import polars as pl
import numpy as np
import argparse
from datetime import datetime

def _print(*args, **kw):
    # Printing time for log recording
    print("[%s]" % (datetime.now()),*args, **kw)

#--------# Arguments #--------#

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process VCF files for regulatory regions.")
parser.add_argument("--chrom", type=int, required=True, help="Chromosome number (1-22)")
chrom = parser.parse_args().chrom

_print(f"Chromosome: {chrom}")

#--------# Directories #--------#

# Change working directory
os.chdir('/pool01/projects/abante_lab/ao_prediction_enrollhd_2024/')

# Data directory
out_dir = "enroll_hd/regulatory_vcfs/"
data_dir = "enroll_hd/vcfs/"

# Cis elements dataset
cis_regions_path = "genes/gene_cisreg.txt"

vcf_file = f"{data_dir}gwa12345.mis4.9064.hg38.chr{chrom}.vcf"
out_file = f"{out_dir}gwa12345.mis4.9064.hg38.cisreg0.5.maf0.01.chr{chrom}.vcf"

#--------# Open files #--------#

# Load regulatory elements dataset
cis_regions = pl.read_csv(cis_regions_path, separator='\t')

# Filter for the specified chromosome
cis_regions = cis_regions.filter(pl.col("chrom") == f"chr{chrom}")

# Filter for enhancer score
cis_regions = cis_regions.filter(~((pl.col("feature name") == "Enhancer") & (pl.col("score") < 0.5)))

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
vcf_df = vcf_df.filter(pl.col("CHROM") == 'chr' + str(chrom))

#%% Get MAF mask

# select all columns starting with 'column_'
genotype_cols = [col for col in vcf_df.columns if col.startswith("column_")]

# Define the set of alternate genotypes
alt_genotypes = {"0/1", "1/0", "1/1"}

# Count the occurrences of the alternate genotypes per row
vcf_df = vcf_df.with_columns([
    pl.fold(
        pl.lit(0),  # Initialize with 0
        lambda acc, x: acc + x.is_in(alt_genotypes).cast(pl.Int8),  # Add 1 if genotype is in alt_genotypes
        genotype_cols  # Apply across all genotype columns
    ).alias("alt_genotype_count")
]).with_columns([
    (pl.col("alt_genotype_count") / len(genotype_cols)).alias("alt_genotype_freq") # Normalize by the number of samples
])

# Filter for alt genotype frequency > 1%
vcf_df = vcf_df.filter(pl.col("alt_genotype_freq") > 0.01)

# Drop helper columns
vcf_df = vcf_df.drop(["alt_genotype_count", "alt_genotype_freq"])
#%%
# Start with an empty list to collect filtered chunks
filtered_chunks = []

# For each interval in cis_regions, filter vcf_df accordingly
for row in cis_regions.iter_rows(named=True):

    start = row["start"]
    end = row["end"]
    filtered = vcf_df.filter((pl.col("POS") >= start) & (pl.col("POS") <= end))
    filtered_chunks.append(filtered)

# Concatenate all filtered chunks
vcf_df = pl.concat(filtered_chunks)

# Drop duplicates
vcf_df = vcf_df.unique()

_print(f"Filtered vcf shape: ", vcf_df.shape)
#%%
# Write the output VCF, keeping the original header
with open(out_file, "w") as f:
    f.writelines(header_lines)  # Write the original header
    vcf_df.write_csv(f, separator="\t", include_header=False)
# %%
# # Get the cis region when having a single position
# POS = 23752298
# cis_region = cis_regions.filter((pl.col("start") <= POS) & (pl.col("end") >= POS))