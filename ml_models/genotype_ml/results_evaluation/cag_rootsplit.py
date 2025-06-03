#!/usr/bin/env python3

### At what CAG threshold are genes relevant in all xgboosts ###
#%%
import os
import pickle
import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import sys
import seaborn as sns

os.chdir('/pool01/projects/abante_lab/ao_prediction_enrollhd_2024/')

# Data directory
data_dir = "features/"

out_dir = "ml_results/classification/v2/paperfigs/"

# Path to models
model_dir = "ml_results/classification/v2/xgboosts/"

# Path to output directory
feat_dir = "ml_results/classification/v2/" 

# Read feature contributions
feat_importances = pd.read_csv(f"{feat_dir}b1_xgboost_total_importance.txt", sep="\t")

# Read metrics table
metrics = pd.read_csv(f"{feat_dir}benchmark/combined_performances.txt", sep="\t")

# Read SNP lookup table
snp_lookuptab = pd.read_csv("genes/snps_gene_GO_m3.txt", sep="\t")

# Fill missing GO terms
snp_lookuptab["GO_term"] = snp_lookuptab["GO_term"].fillna("extra_genes")

gene_map = dict(zip(snp_lookuptab['SNP'], snp_lookuptab['Gene']))
go_term_map = dict(zip(snp_lookuptab['SNP'], snp_lookuptab['GO_term']))
gene_to_go_term = dict(zip(snp_lookuptab['Gene'], snp_lookuptab['GO_term']))
#%%
# Extract feature matrix header
# Load header and feature names
with open(data_dir + "subsetting/header_feature_matrix_m3_filt_0.01.txt", "r") as file:
    header = file.readline().strip().split("\t")
feature_names = header[1:]

# Load metrics and filter
valid_metrics = metrics[(metrics["Depth"] == 2) &
                        (metrics["Model Type"] == "b1") &
                        (metrics["Model"] == "XGBoost")]

seeds = valid_metrics["Seed"].unique()
# seeds = np.array([10,15, 16])

# Helper function to unscale CAG
def unscale_cag(cag):
    return cag * (55 - 40) + 40

# Store all records
all_records = []

# Store proportion of f1 CAG trees
tree_props = []

# Iterate through all valid seeds
for seed in seeds:
    tree_file = f"{model_dir}b1_ES_xgboost_{seed}_trees.csv"
    if not os.path.exists(tree_file):
        print(f"Missing trees for seed {seed}")
        continue
    
    trees = pd.read_csv(tree_file, sep="\t", index_col=0)
    f1_roots = trees[(trees['Node'] == 0) & (trees['Feature'] == 'f1')]

    n_trees = len(trees['Tree'].unique())
    n_f1cag_trees = len(f1_roots['Tree'].unique())
    tree_props.append({'Seed':seed, 'N_trees': n_trees, 'f1CAG_trees': n_f1cag_trees, 'proportion': round((n_f1cag_trees/n_trees*100), 3)})

    for _, root in f1_roots.iterrows():
        tree_id = root['Tree']
        root_split = unscale_cag(float(root['Split']))

        second_nodes = trees[
            (trees['Tree'] == tree_id) &
            (trees['ID'].isin([root['Yes'], root['No'], root['Missing']])) &
            (trees['Feature'] != 'Leaf')
        ]

        for _, second in second_nodes.iterrows():
            second_feature = feature_names[int(second['Feature'][1:])]
            second_split = float(second['Split'])
            second_id = second['ID']

            leaf_nodes = trees[
                (trees['Tree'] == tree_id) &
                (trees['ID'].isin([second['Yes'], second['No'], second['Missing']])) &
                (trees['Feature'] == 'Leaf')
            ]

            for _, leaf in leaf_nodes.iterrows():
                all_records.append({
                    'Tree': tree_id,
                    'Seed': seed,
                    'RootSplit': root_split,
                    'SecondNodeID': second_id,
                    'SecondFeature': second_feature,
                    'Gene': gene_map.get(second_feature, 'Unknown'),
                    'SecondSplit': second_split,
                    'LeafNodeID': leaf['ID'],
                    'Prediction': leaf['Gain']    
                })

# Create aggregated DataFrame
secondary_split_summary = pd.DataFrame(all_records).drop_duplicates(subset=["SecondNodeID", "Seed"])

# Load top features from global importance
top_genes = secondary_split_summary['Gene'].value_counts().head(20).index

def rootcag_col_histogram(secondary_split_summary, top_genes):
    """
    Plot a heatmap where each column is a gene and the rows are different root CAG splits.
    The value in the cells corresponds to how many trees are using that gene as a 
    secondary split, having CAG at the root node with a particular split value.
    """
    filtered_df = secondary_split_summary[secondary_split_summary['Gene'].isin(top_genes)]

    # Create pivot table: RootSplit as rows, Gene as columns
    pivot = pd.pivot_table(filtered_df, values='Tree', index='RootSplit', columns='Gene', aggfunc='count', fill_value=0)

    # Histogram column: total counts per RootSplit (row-wise sum)
    pivot.insert(0, 'All', pivot.sum(axis=1))

    # Sort gene columns (excluding 'All') by lowest RootSplit they appear in
    def first_nonzero_index(col):
        return next((i for i, val in enumerate(col) if val > 0), float('inf'))

    gene_order = sorted([gene for gene in pivot.columns if gene != 'All'],
                        key=lambda gene: first_nonzero_index(pivot[gene].values))

    # Reinsert 'All' as the first column
    pivot = pivot[['All'] + gene_order]

    # Plot with matplotlib
    fig, ax = plt.subplots(figsize=(14, 8))
    # Apply logarithmic normalization to color scale
    im = ax.imshow(pivot.values, cmap='Blues', norm=PowerNorm(gamma=0.5))
    # Axis ticks and labels
    ax.set_xticks(np.arange(pivot.columns.shape[0]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")

    ax.set_yticks(np.arange(pivot.index.shape[0]))
    ax.set_yticklabels([int(r) for r in pivot.index])  # Ensure y-axis is integer labels

    # Annotate each cell with its value
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center", color="black", fontsize=12)

    # Titles and labels
    ax.set_title("RootSplit vs Top 20 Genes Highest Gain in 2nd Split")
    ax.set_xlabel("Gene (and Total)")
    ax.set_ylabel("CAG RootSplit")

    fig.tight_layout()
    # plt.show()

    return fig, ax

# Now use the existing function to plot
fig, ax = rootcag_col_histogram(secondary_split_summary, top_genes)
# %%
secondary_split_summary.to_csv(f"{feat_dir}secondary_splits_allxgb1.txt", sep='\t', index=False)
fig.savefig(f"{out_dir}rootsplit_vs_2ndfeat_all.pdf", dpi=300)

#%%
tree_props_df = pd.DataFrame(tree_props)
tree_props_df.to_csv(f"{feat_dir}f1cag_prop_allxgb1.txt", sep='\t', index=False)
# %%
tree_props_df['proportion'].mean()

#%% For b2

# Load metrics and filter
valid_metrics = metrics[(metrics["Depth"] == 2) &
                        (metrics["Model Type"] == "b2") &
                        (metrics["Model"] == "XGBoost")]

seeds = valid_metrics["Seed"].unique()

# Store proportion of f1 CAG trees
tree_props = []

# Iterate through all valid seeds
for seed in seeds:
    tree_file = f"{model_dir}b1_ES_xgboost_{seed}_trees.csv"
    if not os.path.exists(tree_file):
        print(f"Missing trees for seed {seed}")
        continue
    
    trees = pd.read_csv(tree_file, sep="\t", index_col=0)
    f1_roots = trees[(trees['Node'] == 0) & (trees['Feature'] == 'f1')]

    n_trees = len(trees['Tree'].unique())
    n_f1cag_trees = len(f1_roots['Tree'].unique())
    tree_props.append({'Seed':seed, 'N_trees': n_trees, 'f1CAG_trees': n_f1cag_trees, 'proportion': round((n_f1cag_trees/n_trees*100), 3)})

tree_props_df = pd.DataFrame(tree_props)
tree_props_df.to_csv(f"{feat_dir}f1cag_prop_allxgb2.txt", sep='\t', index=False)

tree_props_df['proportion'].mean()