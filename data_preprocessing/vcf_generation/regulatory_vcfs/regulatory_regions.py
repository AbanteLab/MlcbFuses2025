#!/usr/bin/env python3

### Produces file with regulatory regions of our genes ###
#%%
# Dependencies (conda environment: borzoi_py310)
import os 
import pandas as pd
import polars as pl
import numpy as np
from gtfparse import read_gtf

#--------# Directories #--------#

# Change working directory
os.chdir('/pool01/projects/abante_lab/ao_prediction_enrollhd_2024/')

# Data directory
data_dir = "enroll_hd/vcfs/"
out_dir = "genes/"

# Cis elements dataset
genehancer_path = "genes/genehancer.txt"

# TSS dataset
tss_path = "genes/gencode.v47.basic.annotation.gtf"

# Gene path
gene_path = "genes/snps_gene_GO_m3.txt"

# Genes
genes_path = "features/gene_tensor_order_nonrep.txt"

#--------# Regulatory vcfs #--------#

# Load genehancer dataset
genehancer = pd.read_csv(genehancer_path, sep='\t')

# Load TSS dataset
tss_df = read_gtf(tss_path)
# Filter for transcripts
tss_df = tss_df.filter(tss_df["feature"] == "transcript")
# Filter: chromosomes named '1' to '22', 'X', or 'Y'
tss_df = tss_df.filter(pl.col("seqname").is_in(['chr' + str(i) for i in range(1, 23)] + ["X", "Y"]))
# Compute TSS using conditional logic
tss_df = tss_df.with_columns([
    pl.when(pl.col("strand") == "+")
      .then(pl.col("start"))
      .otherwise(pl.col("end"))
      .alias("TSS")
])
# Drop all columns except 'Gene', 'TSS', and 'Strand'
tss_df = tss_df.select(["gene_name", "TSS", "strand"])

# Load list of genes
genes_df = pd.read_csv(genes_path, sep='\t', header=None)
genes = genes_df[0].tolist()

# Load lookup table
gene_lookup = pd.read_csv(gene_path, sep='\t')

# Create empty dataset
cis_regions = pd.DataFrame(columns=['chrom', 'target_gene', 'source', 'feature name', 'start', 'end', 'score', 'strand', 'frame', 'attributes', 'genehancer_id', 'connected_genes'])

for gene in genes:  

    # Filter elements conntected to gene
    cis_gene = genehancer[genehancer['connected_genes'].apply(lambda x: gene in x)]

    if gene not in tss_df['gene_name'].to_list():
        # There are two genes with different names in the TSS dataset
        if gene == 'USP41':
            gene_alt = 'ENSG00000266470'
            gene_tss = tss_df.filter(pl.col('gene_name') == gene_alt).row(0)
        elif gene == 'LRRC29':
            gene_alt = 'FBXL9P'
            gene_tss = tss_df.filter(pl.col('gene_name') == gene_alt).row(0)
        else:
            # If the gene is not found in the TSS dataset, skip it
            print(f"Gene {gene} not found in TSS dataset. Skipping...")
            continue
    else:
        gene_tss = tss_df.filter(pl.col('gene_name') == gene).row(0)
    
    # Add row representing promoter (-2kbp + 2kbp)
    new_row = {
        'chrom': 'chr'+ str(gene_lookup[gene_lookup['Gene'] == gene]['chromosome'].values[0]),
        'source': 'gencodev47',
        'feature name': 'Promoter',
        'start': int(gene_tss[1]) - 2000,
        'end': int(gene_tss[1]) + 2000,
        'score': np.nan,    
        'strand': gene_tss[2],
        'frame': np.nan,
        'attributes': np.nan,
        'genehancer_id': 'Promoter_' + gene,
        'connected_genes': {gene: 1.0}
    }

    # Append the new row to the DataFrame
    cis_gene = pd.concat([cis_gene, pd.DataFrame([new_row])], ignore_index=True)

    # Add gene name to cis_gene
    cis_gene['target_gene'] = gene

    # Append to cis_regions
    cis_regions = pd.concat([cis_regions, cis_gene], ignore_index=True)

# Save cis_regions to file
cis_regions.to_csv("genes/gene_cisreg.txt", sep='\t', index=False)