#!/usr/bin/env python3

### Assembles vcf files with SNPs from regulatory regions of our genes ###
#%%
# Dependencies (conda environment: enrollhd)
import os 
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt

chrom = 18

#--------# Directories #--------#

# Change working directory
os.chdir('/pool01/projects/abante_lab/ao_prediction_enrollhd_2024/')

# Data directory
data_dir = "enroll_hd/regulatory_vcfs/"

chroms_alt = []

for chrom in range(1, 23):
    vcf_file = f"{data_dir}gwa12345.mis4.9064.hg38.chr{chrom}.cisreg.vcf"

    #--------# Open files #--------#

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
    ])

    alt_counts = vcf_df["alt_genotype_count"]

    # If you want the result as a Python list (vector), you can use:
    vector = alt_counts.to_list()
    vector = [v/9064 for v in vector]  # Normalize by the number of samples

    # plt.hist(vector, bins=50)
    # plt.title(f"Histogram of Alt Genotype Counts for Chromosome {chrom}")
    # plt.show()

    maf001 = [v for v in vector if v > 0.01]
    maf002 = [v for v in vector if v > 0.02]
    maf005 = [v for v in vector if v > 0.05]
    maf010 = [v for v in vector if v > 0.10]
    maf020 = [v for v in vector if v > 0.20]
    

    chroms_alt.append([chrom, len(vector), len(maf001), len(maf002), len(maf005), len(maf010), len(maf020)])
# %%
chroms_alt = pd.DataFrame(chroms_alt, columns=['chrom', 'num_snps', 'num_snps_maf_0.01', 'num_snps_maf_0.02', 'num_snps_maf_0.05', 'num_snps_maf_0.10', 'num_snps_maf_0.20'])
chroms_alt.to_csv(f"{data_dir}maf_counts.txt", sep="\t", index=False)


#%% Read results table
chroms_alt = pd.read_csv(f"{data_dir}maf_counts.txt", sep="\t")

chroms_alt_sum = chroms_alt[['num_snps', 'num_snps_maf_0.01']].sum()
chroms_alt_sum['total_reduction'] = chroms_alt_sum['num_snps_maf_0.01'] / chroms_alt_sum['num_snps'] * 100

# %%
# For each num_snps_maf_x, calculate the percentage of SNPs with MAF > x
chroms_alt_percentages = chroms_alt[['chrom']].copy()
chroms_alt_percentages['percent_snps_maf_0.01'] = (chroms_alt['num_snps_maf_0.01'] / chroms_alt['num_snps'] * 100).round(2)
chroms_alt_percentages['percent_snps_maf_0.02'] = (chroms_alt['num_snps_maf_0.02'] / chroms_alt['num_snps'] * 100).round(2)
chroms_alt_percentages['percent_snps_maf_0.05'] = (chroms_alt['num_snps_maf_0.05'] / chroms_alt['num_snps'] * 100).round(2)
chroms_alt_percentages['percent_snps_maf_0.10'] = (chroms_alt['num_snps_maf_0.10'] / chroms_alt['num_snps'] * 100).round(2)
chroms_alt_percentages['percent_snps_maf_0.20'] = (chroms_alt['num_snps_maf_0.20'] / chroms_alt['num_snps'] * 100).round(2)

chroms_alt
# %%
# calculate the mean of each column
mean_values = chroms_alt_percentages.mean().round(2)
mean_values
print('Hours to run all SNPs at 120s/SNP:')
print((chroms_alt['num_snps'].sum()/22)*120/3600/24)
# %%
# Filtering by enhancer scores
