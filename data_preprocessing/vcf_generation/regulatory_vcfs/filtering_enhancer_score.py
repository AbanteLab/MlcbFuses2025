#!/usr/bin/env python3

### Produces file with regulatory regions of our genes ###
#%%
# Dependencies (conda environment: borzoi_py310)
import os 
import pandas as pd
import polars as pl
import numpy as np
from gtfparse import read_gtf
from datetime import datetime
import matplotlib.pyplot as plt

def _print(*args, **kw):
    # Printing time for log recording
    print("[%s]" % (datetime.now()),*args, **kw)

#--------# Directories #--------#

# Change working directory
os.chdir('/pool01/projects/abante_lab/ao_prediction_enrollhd_2024/')

# Data directory
data_dir = "enroll_hd/regulatory_vcfs/"
# Cis elements dataset
cis_regions_path = "genes/gene_cisreg.txt"

# Load regulatory elements dataset
cis_regions = pl.read_csv(cis_regions_path, separator='\t')

# Plot istribution of enhancer scores
plt.hist(cis_regions.filter(pl.col("feature name") == "Enhancer")['score'], bins=30)
plt.title("Distribution of Enhancer Scores")
plt.xlabel("Enhancer Score")
plt.ylabel("Frequency")
plt.show()
#%%
# Filter for enhancer score
cis_regions = cis_regions.filter(~((pl.col("feature name") == "Enhancer") & (pl.col("score") < 0.5)))

chroms_alt = []

for chrom in range(1, 23):
    _print(f"chromosome {chrom}")

    vcf_file = f"{data_dir}gwa12345.mis4.9064.hg38.chr{chrom}.cisreg.vcf"

    # Filter for the specified chromosome
    cis_regions_chrom = cis_regions.filter(pl.col("chrom") == f"chr{chrom}")

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
        "column_8": "INFO"
    })

    vcf_df = vcf_df.with_columns(pl.col("POS").cast(pl.Int64))

    # Filter vcf_df for the current chromosome
    vcf_df = vcf_df.filter(pl.col("CHROM") == 'chr' + str(chrom))

    # extract positions from vcf_df
    vcf_positions = vcf_df.select(pl.col("POS")).to_numpy().flatten()

    # For each interval in cis_regions, filter vcf_df accordingly
    filtered_positions = []
    for row in cis_regions_chrom.iter_rows(named=True):
        start = row["start"]
        end = row["end"]
        matches = vcf_positions[(vcf_positions >= start) & (vcf_positions <= end)]
        filtered_positions.extend(matches.tolist())

    # Remove duplicates
    filtered_positions = list(set(filtered_positions))
    _print(f"Filtered vcf shape: ", len(filtered_positions))

    reduction = len(filtered_positions) / len(vcf_positions) * 100

    chroms_alt.append([chrom, len(vcf_positions), len(filtered_positions), round(reduction, 2)])

chroms_alt = pd.DataFrame(chroms_alt, columns=['chrom', 'og_num_variants', 'num_variants', 'reduction'])

# Add a row representing the sum of columns

chroms_alt.to_csv(f"{data_dir}enhancer0.5.txt", sep="\t", index=False)

# sum of columns
chroms_alt_sum = chroms_alt[['og_num_variants', 'num_variants']].sum()
chroms_alt_sum['total_reduction'] = chroms_alt_sum['num_variants'] / chroms_alt_sum['og_num_variants'] * 100

#%% Read chroms_alt
chroms_alt = pd.read_csv(f"{data_dir}enhancer0.5.txt", sep="\t")

# #%%
# # Write the output VCF, keeping the original header
# with open(out_file, "w") as f:
#     f.writelines(header_lines)  # Write the original header
#     filtered_vcf.write_csv(f, separator="\t", include_header=False)