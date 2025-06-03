#!/usr/bin/env python3

#%%## score_variants output rearrangement ###
# run with enrollhd conda environment

import h5py
import pandas as pd
import argparse

# Path to save results
outdir = "/pool01/projects/abante_lab/genomic_llms/borzoi/proc_results/sed_all_chr/"

# Results file name
parser = argparse.ArgumentParser(description="Process Borzoi SED output for a given chromosome.")
parser.add_argument("--result_name", required=True, help="Name of the result file (e.g., chr18)")
args = parser.parse_args()
result_name = args.result_name

# Path of borzoi output"
output_path = f"/pool01/projects/abante_lab/genomic_llms/borzoi/proc_results/sed_all_chr/sed_{result_name}.h5"

#%%## Open the .h5 file ###
sed_h5 = h5py.File(output_path,'r')

# Extract variables
snp_list = [s.decode() for s in sed_h5['snp']]
gene_list = [g.decode() for g in sed_h5['gene']]
snp_idx = sed_h5['si'][:]
logseds = sed_h5['logSED'][:]
target_labels = [t.decode() for t in sed_h5['target_labels']]
alt_alleles = [a.decode() for a in sed_h5['alt_allele']]

results = []

for row_ix, snp_gene_idx in enumerate(snp_idx):
    snp = snp_list[snp_gene_idx]
    gene = gene_list[row_ix]
    alt_allele = alt_alleles[snp_gene_idx]  
    for target_ix, target_label in enumerate(target_labels):
        results.append({
            "snp": snp,
            "gene": gene,
            "alt_allele": alt_allele,
            "tissue": target_label,
            "logSED": logseds[row_ix, target_ix]
        })

# Convert results to a pandas DataFrame
df = pd.DataFrame(results)

# Save the DataFrame to a tab separated file
# Save as compressed TSV
df.to_csv(f"{outdir}sed_{result_name}.txt.gz", sep="\t", index=False, compression="gzip")
# %%
