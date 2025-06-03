#!/usr/bin/env python3

### Aggregate Borzoi SED output for all samples in a given chromosome and tissue ###

# seq 1 22 | xargs -P 5 -I{} python expression_vectors.py --chrom {} --tissue_folder putamen
# seq 1 22 | xargs -P 5 -I{} python expression_vectors.py --chrom {} --tissue_folder caudate

#%%
import polars as pl
import pandas as pd
import argparse
import os
from datetime import datetime

parser = argparse.ArgumentParser(description="Process Borzoi SED output for a given chromosome.")
parser.add_argument("--chrom", required=True, help="Number of chromosome (e.g., 18)")
parser.add_argument("--tissue_folder", required=True, help="Name of tissue folder (putamen or caudate)")
args = parser.parse_args()
chrom = args.chrom
tissue_folder = args.tissue_folder

def _print(*args, **kw):
    # Printing time for log recording
    print("[%s]" % (datetime.now()),*args, **kw)

# chrom = 10  # For testing
# tissue_folder = "putamen"  # For testing
# tissue = "RNA-seq: Putamen"  # For testing

_print(f'Parameters: chrom = {chrom}, tissue = {tissue_folder}')

if tissue_folder == 'putamen':
    tissue = "RNA-seq: Putamen"
elif tissue_folder == 'caudate':
    tissue = "RNA-seq: Caudate"

# Set working directory
os.chdir("/pool01/projects/abante_lab")

out_dir = f"genomic_llms/borzoi/proc_results/expression_vectors/{tissue_folder}/"

# File Paths
vcf_path = f"ao_prediction_enrollhd_2024/enroll_hd/regulatory_vcfs/gwa12345.mis4.9064.hg38.cisreg0.5.mad0.01.chr{chrom}.vcf" 
results_path = f"genomic_llms/borzoi/proc_results/weighted_logSED/{tissue_folder}/filtered_weighted_logSED_chr{chrom}.tsv.gz" 

# Load VCF 
vcf = pl.read_csv(vcf_path, separator="\t", has_header=True, comment_prefix="##")
vcf = vcf.rename({"#CHROM": "CHROM"})

# Identify sample columns
meta_cols = {"CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"}
sample_cols = [col for col in vcf.columns if col not in meta_cols]

# Melt to long format: each row is (snp, sample, genotype)
long_vcf = vcf.select(["ID"] + sample_cols).melt(id_vars="ID", variable_name="sample", value_name="genotype")

# Assign numeric weights
long_vcf = long_vcf.with_columns([
    pl.when(pl.col("genotype") == "1/1").then(pl.lit(2)) # homozygous alternative
      .when(pl.col("genotype").is_in(["0/1", "1/0"])).then(pl.lit(1)) # heterozygous
      .when(pl.col("genotype") == "0/0").then(pl.lit(0)) # homozygous reference
      .otherwise(pl.lit(0))
      .alias("geno_weight")
])
# Read predictions
pred_pl = pl.read_csv(results_path, separator="\t")

# Ensure predictions are from tissue
if pred_pl["tissue"].unique().to_list() != [tissue]:
    raise ValueError(f"Predictions are not for the specified tissue: {tissue}")

# Drop unnecessary columns
pred_pl = pred_pl.drop(["alt_allele", "tissue", "chrom", "logSED", "pos", "genehancer_id"])

# Join VCF variants with predictions on SNP ID
vcf_pred_joined = long_vcf.join(pred_pl, left_on="ID", right_on="snp", how="inner")

# Calculate weighted logSED accounting for both chromosome copies
vcf_pred_joined = vcf_pred_joined.with_columns(
    (pl.col("geno_weight") * pl.col("weighted_logSED")).alias("weighted_logSED_weighted")
)

# Aggregate weighted logSED sums by sample and gene
gene_sums = vcf_pred_joined.group_by(["sample", "gene"]).agg([
    pl.sum("weighted_logSED_weighted").alias("weighted_logSED")
])

# Create expression matrix
matrix_df = (
    gene_sums
    .pivot(index="sample", columns="gene", values="weighted_logSED")
    .fill_null(0.0)
    .sort("sample")  # Alphabetical sort by sample name (like original vcf)
)

matrix_df.write_csv(f"{out_dir}{tissue_folder}_expression_matrix_chr{chrom}.txt", separator="\t")

_print('Results saved for chromosome', chrom, 'in tissue', tissue_folder)