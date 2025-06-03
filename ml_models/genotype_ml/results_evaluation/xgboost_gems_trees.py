#!/usr/bin/env python3

#%%### XGBoost tree structure ###

import os
import pickle
import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import sys
from xgboost import plot_tree
import seaborn as sns
from scipy.stats import fisher_exact
import math
from matplotlib.patches import Patch


PROJECT_ROOT = "/pool01/code/projects/abante_lab/ao_prediction_enrollhd_2024/ml_models"
# PROJECT_ROOT = "/gpfs/projects/ub212/ao_prediction_enrollhd_2024/code/src/ml_models"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.data_loading import _print, load_X_y

os.chdir('/pool01/projects/abante_lab/ao_prediction_enrollhd_2024/')

# Data directory
data_dir = "features/"

# Output directory
out_dir = "ml_results/classification/v2/best_models/tree_structure/"

# Path to models
model_dir = "ml_results/classification/v2/xgboosts/"

# Path to output directory
feat_dir = "ml_results/classification/v2/" 

# Read feature contributions
feat_importances = pd.read_csv(f"{feat_dir}b1_xgboost_total_importance.txt", sep="\t")

# Read metrics table
metrics = pd.read_csv(f"{feat_dir}xgboost_metrics_summary.txt", sep="\t")

# Read SNP lookup table
snp_lookuptab = pd.read_csv("genes/snps_gene_GO_m3.txt", sep="\t")

X_path = data_dir + "scag_feature_matrix_m3_filt_0.01.npz"
y_path = data_dir + "binned_ao.txt"

#%% Function to unscale CAG values
def unscale_cag(cag):
    return cag * (55-40) + 40

#%% b1 trees

# Extract feature matrix header
with open(data_dir + "subsetting/header_feature_matrix_m3_filt_0.01.txt", "r") as file:
    header = file.readline().strip().split("\t")
feature_names = header[1:]
#%%
# Load the model
model_path = f"{model_dir}b1_ES_xgboost_35_model.json"
xgb_booster = xgb.Booster()
xgb_booster.load_model(model_path)

# Load trees
model_trees = f"{model_dir}b1_ES_xgboost_35_trees.csv"
trees = pd.read_csv(model_trees, sep="\t", index_col=0)

#%%
def plot_cag(split_values, feature = None, fig=None, ax=None):
    """
    Plot histogram of split values and return the plot object.
    """
    split_values = split_values.astype(int)  # Round split values to integers

    bins = np.arange(40, 56)  # Define bins from 40 to 55
    hist, bin_edges = np.histogram(split_values, bins=bins)
    if ax is None and fig is None:
            fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(bin_edges[:-1], hist, width=1, edgecolor='black', color='skyblue', align='edge')
    ax.set_xticks(np.arange(40, 56, 1))  # Set x-axis labels from 40 to 55
    ax.set_xlabel('Split Threshold for CAG')
    ax.set_ylabel('Frequency')
    if feature:
        ax.set_title(f"Histogram of CAG Split Values at Root Nodes for {feature}")
    else:
        ax.set_title("Histogram of CAG Split Values at Root Nodes")
    ax.grid(True)
    plt.tight_layout()
    
    return fig, ax

# Filter root nodes (Node == "0")
root_nodes = trees[trees['Node'] == 0]

# Filter those that use feature 'f1'
f1_root_nodes = root_nodes[root_nodes['Feature'] == 'f1']

# How many boosters use f1 as the first split?
num_f1_root_splits = f1_root_nodes.shape[0]
print(f"Number of boosters using 'f1' as root split: {num_f1_root_splits}")

# Histogram of Split values where f1 is used in the root node
split_values = f1_root_nodes['Split'].astype(float)
split_values = unscale_cag(split_values)
plot_cag(split_values)

#%%
# Filter root nodes where the root feature is f1
f1_roots = trees[(trees['Node'] == 0) & (trees['Feature'] == 'f1')].copy()

# Initialize list to collect second-level feature info
records = []

# Loop through f1-rooted trees
for _, row in f1_roots.iterrows():
    tree_id = row['Tree']
    root_split = float(row['Split'])
    yes_node = row['Yes']
    no_node = row['No']
    missing_node = row['Missing']
    
    # Find all children of the root node in the same tree
    child_nodes = trees[(trees['Tree'] == tree_id) & (trees['ID'].isin([yes_node, no_node, missing_node]))]
    
    # For each child, record its feature and the root split
    for _, child in child_nodes.iterrows():
        feature = child['Feature']
        if feature != 'Leaf':  # Only count splits
            records.append({'RootSplit': unscale_cag(root_split), 'SecondFeature': feature_names[int(feature[1:])]})

# Create DataFrame from records
split_df = pd.DataFrame(records)

# Bin the root split values
split_df['RootBin'] = split_df['RootSplit'].astype(int)

# Group and calculate proportions
counts = split_df.groupby(['RootBin', 'SecondFeature']).size().reset_index(name='Count')

# Get total counts per RootBin
counts['Proportion'] = counts.groupby('RootBin')['Count'].transform(lambda x: x / x.sum())

# Map 'SecondFeature' to 'Gene' and 'GO_term' using the SNP lookuptab
gene_map = dict(zip(snp_lookuptab['SNP'], snp_lookuptab['Gene']))
go_term_map = dict(zip(snp_lookuptab['SNP'], snp_lookuptab['GO_term']))

counts['Gene'] = counts['SecondFeature'].map(gene_map)
counts['GO_term'] = counts['SecondFeature'].map(go_term_map)

# Get most common second features for each root bin
most_common_features = counts.groupby('RootBin').apply(lambda x: x.nlargest(3, 'Count')).reset_index(drop=True)
# %%
records = []

# Get all trees where root node is 'f1'
f1_roots = trees[(trees['Node'] == 0) & (trees['Feature'] == 'f1')]

for _, root in f1_roots.iterrows():
    tree_id = root['Tree']
    root_split = unscale_cag(float(root['Split']))

    # Get the IDs of the root's children (second-level nodes)
    second_nodes = trees[
        (trees['Tree'] == tree_id) &
        (trees['ID'].isin([root['Yes'], root['No'], root['Missing']])) &
        (trees['Feature'] != 'Leaf')  # Must be split nodes
    ]

    for _, second in second_nodes.iterrows():
        second_feature = feature_names[int(second['Feature'][1:])]
        second_split = float(second['Split'])
        second_id = second['ID']

        # Get the children of this second-level node (leaves at depth 2)
        leaf_nodes = trees[
            (trees['Tree'] == tree_id) &
            (trees['ID'].isin([second['Yes'], second['No'], second['Missing']])) &
            (trees['Feature'] == 'Leaf')
        ]

        for _, leaf in leaf_nodes.iterrows():
            records.append({
                'Tree': tree_id,
                'RootSplit': root_split,
                'SecondNodeID': second_id,
                'SecondFeature': second_feature,
                'Gene': gene_map.get(second_feature, 'Unknown'),
                'GO': snp_lookuptab[snp_lookuptab['SNP']==second_feature]['GO_term'].values[0],
                'SecondSplit': second_split,
                'LeafNodeID': leaf['ID'],
                'Prediction': leaf['Gain']
            })

# Convert to DataFrame
secondary_split_summary = pd.DataFrame(records)
secondary_split_summary[secondary_split_summary['Gene']=='FAN1']

#%%
# Get 3 most common second features
most_common_second_features = (
    secondary_split_summary['SecondFeature']
    .value_counts()
    .head(16)
    .reset_index()
)
print(most_common_second_features)

#%%
# Top 20 genes by frequency
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
    im = ax.imshow(pivot.values, cmap='Blues')

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
                ax.text(j, i, str(val), ha="center", va="center", color="black", fontsize=8)

    # Titles and labels
    ax.set_title("RootSplit vs Top 20 Genes Most Used in 2nd Split")
    ax.set_xlabel("Gene (and Total)")
    ax.set_ylabel("CAG RootSplit")

    fig.tight_layout()
    # plt.show()

    return fig, ax


# %%
plt.figure(figsize=(30, 20))  
plot_tree(xgb_booster, num_trees=369) 
plt.show()

# %% Which secondary features are used in the left and right branches (smaller or larger than splitting value)

results = []

# For each tree with f1 root
for _, root in f1_roots.iterrows():
    tree_id = root['Tree']
    root_split = root['Split']
    
    yes_node = root['Yes']
    yes_node = int(yes_node.split('-')[-1])
    no_node = root['No']
    no_node = int(no_node.split('-')[-1])
    
    # Get subtree rows for this tree
    subtree = trees[trees['Tree'] == tree_id]

    # Analyze the "Yes" branch
    yes_subnode = subtree[subtree['Node'] == yes_node]
    if not yes_subnode.empty:
        yes_subnode = yes_subnode.iloc[0]
        second_feature = yes_subnode['Feature']
        second_feature_num = int(second_feature[1:])  # assuming format 'f123'
        snp = feature_names[second_feature_num]
        go = snp_lookuptab[snp_lookuptab['SNP']==snp]['GO_term'].values[0]
        gene = gene_map.get(feature_names[second_feature_num], 'Unknown')
        results.append({
            'Tree': tree_id,
            'Root_Split': int(unscale_cag(root_split)),
            'Branch': 'Smaller',  # means f1 <= split
            'Subnode_ID': yes_node,
            'Subnode_Feature': gene,
            'Subnode_SNP': snp,
            'GO': go
            # 'Subnode_Split': yes_subnode['Split']
        })
    
    # Analyze the "No" branch
    no_subnode = subtree[subtree['Node'] == no_node]
    if not no_subnode.empty:
        no_subnode = no_subnode.iloc[0]
        second_feature = no_subnode['Feature']
        second_feature_num = int(second_feature[1:])  # assuming format 'f123'
        snp = feature_names[second_feature_num]
        go = snp_lookuptab[snp_lookuptab['SNP']==snp]['GO_term'].values[0]
        gene = gene_map.get(feature_names[second_feature_num], 'Unknown')
        results.append({
            'Tree': tree_id,
            'Root_Split': int(unscale_cag(root_split)),
            'Branch': 'Larger',   # means f1 > split
            'Subnode_ID': no_node,
            'Subnode_Feature': gene,
            'Subnode_SNP': snp,
            'GO': go
            # 'Subnode_Split': no_subnode['Split']
        })

final_df = pd.DataFrame(results)
print(final_df)
# %%
results_filt = final_df[final_df['Subnode_Feature'].isin(top_genes)]

# Count Branch occurrences per Subnode_Feature
counts = final_df.groupby(['Subnode_Feature', 'Branch']).size().unstack(fill_value=0)

# Plot
counts.plot(kind='bar', figsize=(18, 6))
plt.ylabel('# Trees')
plt.title('Branch Counts per Secondary Features')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% 
GO_names = {'GO:0006298': 'DNA maintenance',
            'GO:0035249': 'Glutamatergic synaptic transmission',
            'GO:0008242': 'Omega peptidase activity',
            'GO:0004197': 'Cysteine-type endopeptidase activity',
            'GO:0043161': 'Ubiquitin-dependant protein catabolic process',
            'GO:0043130': 'Ubiquitin binding',
            'GO:0031625': 'Ubiquitin protein ligase binding',
            'GO:0016579': 'Protein deubiquitination',
            'GO:0140110': 'Transcription regulator activity',
            'GO:0051402': 'Neuron apoptotic process',
            'GO:0042157': 'Lipoprotein metabolic process',
            'GO:0098930': 'Axonal transport',
            'GO:0046655': 'Folic acid metabolism',
            'GO:0006112': 'Energy reserve metabolism'}

# Convert snps_feature_names to a df
snps_df = pd.DataFrame(feature_names, columns=['SNP'])

# Drop duplicates from snp_lookuptab based on the SNP column
snp_lookuptab_unique = snp_lookuptab.drop_duplicates(subset=['SNP'])

# Merge snps_df with the deduplicated snp_lookuptab on the SNP column
model_snps = snps_df.merge(snp_lookuptab_unique, on='SNP', how='left')

# Rename GO column
model_snps = model_snps.rename(columns={'GO_term': 'GO'})

# Drop columns from CAG and sex
model_snps = model_snps[~model_snps["SNP"].isin(["CAG", "Sex"])]

background = model_snps['GO'].value_counts()

distributions = []

# Append background GO percentages
distributions.append(background)

# Calculate the percentage distribution of GO categories in the top20 of the best model

go_counts_smaller = final_df[final_df['Branch']=='Smaller']['GO'].value_counts()
distributions.append(go_counts_smaller)
go_counts_larger = final_df[final_df['Branch']=='Larger']['GO'].value_counts()
distributions.append(go_counts_larger)

model_names_bg = ['Background', "smaller", "larger"]

# Combine distributions into a single DataFrame
combined_df = pd.DataFrame(distributions).T.fillna(0)
combined_df.columns = model_names_bg

# Add 1 to all values to avoid having 0 counts
combined_df = combined_df + 1

# Column Total
combined_df_modeltotal = combined_df.sum()

combined_df_modeltotal
# %%
# Perform Fisher's exact test for each GO term against the background
pval_results = {}
odds_results = {}

for go_term in combined_df.index:
    go_proportions = combined_df.loc[go_term].drop('Background')  # Exclude 'Background' column
    background_count = combined_df.loc[go_term, 'Background']
    p_values = {}
    odds = {}
    for model, proportion in go_proportions.items():
        # Contingency table assembly
        contingency_tab = [[proportion, combined_df_modeltotal[model] - proportion], [background_count, combined_df_modeltotal['Background'] - background_count]]
        
        # Fisher test
        oddsratio, p_value = fisher_exact(contingency_tab)
        
        # Save results in corresponding vectors
        p_values[model] = p_value
        odds[model] = oddsratio
        
    # Add result vector to dictionary
    pval_results[go_term] = p_values
    odds_results[go_term] = odds

# Convert results to DataFrame
pval_results_df = pd.DataFrame(pval_results)
odds_results_df = pd.DataFrame(odds_results)

odds_results_df = odds_results_df.round(4).T
result = pval_results_df.round(4).T

# Replace indices with corresponding names
odds_results_df.index = odds_results_df.index.map(GO_names)
result.index = result.index.map(GO_names)

odds_smaller = odds_results_df['smaller']
p_smaller = result['smaller']
odds_larger = odds_results_df['larger']
p_larger = result['larger']
#%%
def plot_enrichment_bar(odds_series, p_series, title='Enrichment analysis', figsize=(8, 5)):
    """
    Plot enrichment bar plot using log10(odds ratios) and p-values.
    
    Parameters:
    - odds_series (pd.Series): Series of odds ratios.
    - p_series (pd.Series): Series of p-values, must align with odds_series.
    - title (str): Title of the plot.
    - figsize (tuple): Figure size.
    
    Returns:
    - fig, ax: Matplotlib figure and axes objects.
    """
    # Calculate log10 of odds ratios
    log_odds = odds_series.apply(lambda x: math.log10(x))
    
    # Combine into a DataFrame
    combined = pd.DataFrame({'odds': log_odds, 'pvals': p_series})
    
    # Sort by absolute value of odds
    combined['abs_odds'] = combined['odds'].abs()
    combined = combined.sort_values(by='abs_odds', ascending=False).drop('abs_odds', axis=1)
    
    # Reverse for plotting (largest at top)
    combined = combined.iloc[::-1]
    
    # Final sorting by actual odds value (negative to positive)
    combined = combined.sort_values(by='odds')
    
    # Determine colors based on p-values
    colors = ['gray' if p < 0.05 else 'lightgray' for p in combined['pvals']]
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(combined.index, combined['odds'], color=colors)
    ax.set_ylabel('log10(Odds Ratio)')
    ax.set_title(title)
    ax.grid(axis='y')
    ax.set_xticks(range(len(combined.index)))
    ax.set_xticklabels(combined.index, rotation=30, ha='right')
    
    # Legend
    legend_elements = [
        Patch(facecolor='gray', edgecolor='black', label='p-value < 0.05'),
        Patch(facecolor='lightgray', edgecolor='black', label='p-value >= 0.05')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    
    return fig, ax

# %%
fig1, ax1 = plot_enrichment_bar(odds_smaller, p_smaller, title='Enrichment Analysis of Secondary Nodes for Smaller CAG Root Splitting')
plt.show()
# %%
fig2, ax2 = plot_enrichment_bar(odds_larger, p_larger, title='Enrichment Analysis of Secondary Nodes for Larger CAG Root Splitting')
plt.show()
# %%
def plot_dual_enrichment_bar(odds_series_1, p_series_1,
                              odds_series_2, p_series_2,
                              titles=('Enrichment 1', 'Enrichment 2'),
                              figsize=(14, 6)):

    def prepare_data(odds, pvals):
        log_odds = odds.apply(lambda x: math.log10(x))
        df = pd.DataFrame({'odds': log_odds, 'pvals': pvals})
        df['abs_odds'] = df['odds'].abs()
        df = df.sort_values(by='abs_odds', ascending=False).drop('abs_odds', axis=1)
        df = df.iloc[::-1]
        df = df.sort_values(by='odds')
        colors = ['gray' if p < 0.05 else 'lightgray' for p in df['pvals']]
        return df, colors

    data1, colors1 = prepare_data(odds_series_1, p_series_1)
    data2, colors2 = prepare_data(odds_series_2, p_series_2)

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    for ax, data, colors, title in zip(axes, [data1, data2], [colors1, colors2], titles):
        ax.bar(data.index, data['odds'], color=colors)
        ax.set_title(title)
        ax.set_ylabel('log10(Odds Ratio)')
        ax.grid(axis='y')
        ax.set_xticks(range(len(data.index)))
        ax.set_xticklabels(data.index, rotation=30, ha='right')

    # Shared legend
    legend_elements = [
        Patch(facecolor='gray', edgecolor='black', label='p-value < 0.05'),
        Patch(facecolor='lightgray', edgecolor='black', label='p-value >= 0.05')
    ]
    axes[0].legend(handles=legend_elements, loc='upper left')
    axes[1].legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()
    return fig, axes# %%

# %%
fig, axes = plot_dual_enrichment_bar(
    odds_smaller, p_smaller,
    odds_larger, p_larger,
    titles=('Smaller CAG', 'Larger CAG')
)
fig.savefig(f"{out_dir}enrichment_secondary.pdf", dpi=300, bbox_inches='tight')

# %% Enrichment for each CAG splitting value
top_genes = secondary_split_summary['Gene'].value_counts().head(20).index
filtered_df = secondary_split_summary[secondary_split_summary['Gene'].isin(top_genes)]

# Convert snps_feature_names to a df
snps_df = pd.DataFrame(feature_names, columns=['SNP'])

# Drop duplicates from snp_lookuptab based on the SNP column
snp_lookuptab_unique = snp_lookuptab.drop_duplicates(subset=['SNP'])

# Merge snps_df with the deduplicated snp_lookuptab on the SNP column
model_snps = snps_df.merge(snp_lookuptab_unique, on='SNP', how='left')

# Rename GO column
model_snps = model_snps.rename(columns={'GO_term': 'GO'})

# Drop columns from CAG and sex
model_snps = model_snps[~model_snps["SNP"].isin(["CAG", "Sex"])]

background = model_snps['GO'].value_counts()

distributions = []

# Append background GO percentages
distributions.append(background)

model_names_bg = ['Background']

# Calculate the percentage distribution of GO categories in the top20 of the best model
for cag_split in filtered_df['RootSplit'].unique():
    go_counts = filtered_df[filtered_df['RootSplit']==cag_split]['GO'].value_counts()
    distributions.append(go_counts)
    model_names_bg.append('cag' + str(int(cag_split)))

# Combine distributions into a single DataFrame
combined_df = pd.DataFrame(distributions).T.fillna(0)
combined_df.columns = model_names_bg

# Add 1 to all values to avoid having 0 counts
combined_df = combined_df + 1

# Column Total
combined_df_modeltotal = combined_df.sum()

combined_df_modeltotal
# %%
# Perform Fisher's exact test for each GO term against the background
pval_results = {}
odds_results = {}

for go_term in combined_df.index:
    go_proportions = combined_df.loc[go_term].drop('Background')  # Exclude 'Background' column
    background_count = combined_df.loc[go_term, 'Background']
    p_values = {}
    odds = {}
    for model, proportion in go_proportions.items():
        # Contingency table assembly
        contingency_tab = [[proportion, combined_df_modeltotal[model] - proportion], [background_count, combined_df_modeltotal['Background'] - background_count]]
        
        # Fisher test
        oddsratio, p_value = fisher_exact(contingency_tab)
        
        # Save results in corresponding vectors
        p_values[model] = p_value
        odds[model] = oddsratio
        
    # Add result vector to dictionary
    pval_results[go_term] = p_values
    odds_results[go_term] = odds

# Convert results to DataFrame
pval_results_df = pd.DataFrame(pval_results)
odds_results_df = pd.DataFrame(odds_results)

odds_results_df = odds_results_df.round(4).T
result = pval_results_df.round(4).T

# Replace indices with corresponding names
odds_results_df.index = odds_results_df.index.map(GO_names)
result.index = result.index.map(GO_names)
#%%
for cag in [46, 44, 45, 43, 42]:
    cag_str = 'cag' + str(cag)
    # Extract columns
    odds = odds_results_df[cag_str]
    p= result[cag_str]

    # Plot enrichment
    fig, ax = plot_enrichment_bar(odds, p, title=f'Enrichment Analysis of Secondary Nodes for CAG root splitting at {cag}')
    fig.savefig(f'{out_dir}enrichment_cag{cag}.pdf', dpi = 300)
    plt.show()

# %%
