#!/usr/bin/env python3

#%%### XGBoost feature contribution aggregation ###

import os
import pickle
import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from itertools import combinations
import itertools
import seaborn as sns
from scipy.stats import wilcoxon

os.chdir('/pool01/projects/abante_lab/ao_prediction_enrollhd_2024/')

# Data directory
data_dir = "features/"

# Path to models
model_dir = "ml_results/classification/v2/xgboosts/"

# Path to output directory
out_dir = "ml_results/classification/v2/" 

# Read SNP lookup table
snp_lookuptab = pd.read_csv("genes/snps_gene_GO_m3.txt", sep="\t")

# Read metrics table
metrics = pd.read_csv(f"{out_dir}xgboost_metrics_summary.txt", sep="\t")

#%% Load b1 models

total_importance = pd.DataFrame(columns=['SNP','Score','Gene','GO', 'Seed'])

# Extract feature matrix header
with open(data_dir + "subsetting/header_feature_matrix_m3_filt_0.01.txt", "r") as file:
    header = file.readline().strip().split("\t")
feature_names = header[1:]

# Loop through all models in the model_dir that start with "b1"
for model_file in os.listdir(model_dir):
    if model_file.startswith("b1_ES_xgboost_") and model_file.endswith("_model.json"):
        # Extract model name from the filename
        seed = model_file.split("_")[3]
        model_name = f"b1_ES_xgboost_{seed}"

        acc = metrics[metrics['Model'] == model_name]['Accuracy'].values[0]
        
        # Load the model
        model_path = os.path.join(model_dir, model_file)
        xgb_booster = xgb.Booster()
        xgb_booster.load_model(model_path)

        gain_importances = xgb_booster.get_score(importance_type='gain')

        gain_importances_list = [[feature_names[int(key[1:])], value] for key, value in gain_importances.items()]

        # Sort in descending order
        gain_importances_list = sorted(gain_importances_list, key=lambda x: x[1], reverse=True)

        # Assemble presenting table
        booster_important_snps = pd.DataFrame(columns=['SNP','Gain','Gene','GO'])

        for snp, gain in gain_importances_list:
            if snp=='CAG':
                continue
            # Find snp in lookup table
            match = snp_lookuptab[snp_lookuptab["SNP"] == snp]
            # Retrieve corresponding gene and GO term
            gene, GOt = match['Gene'].values[0], match['GO_term'].values[0]
            # Create new row in pd df
            row = {'SNP':snp, 'Gain':gain, 'Gene':gene, 'GO':GOt}
            booster_important_snps = booster_important_snps._append(row, ignore_index=True)

        # Add contributions to total_importance table
        for _, row in booster_important_snps.iterrows():
            # Add new row to total_importance
            total_importance = total_importance._append({
                'SNP': row['SNP'],
                'Score': row['Gain'],
                'Gene': row['Gene'],
                'GO': row['GO'],
                'Seed': seed
            }, ignore_index=True)

# Sort in descending order
total_importance = total_importance.sort_values(by='Score', ascending=False).reset_index(drop=True)

#%% Create sets of features for each model seed

def create_seed_sets(total_importance):
    seed_sets = {}

    # Loop through each unique Seed
    for seed_value in total_importance['Seed'].unique():
        # Filter rows for this seed
        subset = total_importance[total_importance['Seed'] == seed_value]
        
        # Sort by Score descending and drop duplicate Genes
        top_genes = subset.sort_values(by='Score', ascending=False)['Gene'].unique()[:20]
        
        # Convert to a set and store
        seed_sets[seed_value] = set(top_genes)
    
    return seed_sets

def jaccard_index(set1, set2):
    return len(set1 & set2) / len(set1 | set2)

#%%
# Group by SNP and compute mean Gain
mean_gain_df = total_importance.groupby('SNP', as_index=False)['Score'].mean()
mean_gain_df = mean_gain_df.rename(columns={'Score': 'Mean_Gain'})

# Drop duplicates from original to get one Gene/GO per SNP (assumes consistency)
snp_metadata = total_importance[['SNP', 'Gene', 'GO']].drop_duplicates(subset='SNP')

# Merge metadata with mean gains
snp_mean_gain_df = pd.merge(mean_gain_df, snp_metadata, on='SNP')

# sort by mean gain
snp_mean_gain_df = snp_mean_gain_df.sort_values(by='Mean_Gain', ascending=False).reset_index(drop=True)

# top 100
b1_top_genes = snp_mean_gain_df['Gene'].unique()[:100]

#%% Load b2 models

total_importance_b2 = pd.DataFrame(columns=['Feature','Gene','Score','GO', 'Seed'])

# get order of features if xgboost VAE encodings
feature_names = pd.read_csv(data_dir + "gene_tensor_order.txt", header=None)
feature_names = pd.concat((pd.DataFrame(['sex', 'CAG']), feature_names))
feature_names = feature_names[0].tolist()

# Loop through all models in the model_dir that start with "b2"
for model_file in os.listdir(model_dir):
    if model_file.startswith("b2_ES_xgboost_") and model_file.endswith("_model.json"):
        # Extract model name from the filename
        seed = model_file.split("_")[3]
        model_name = f"b2_ES_xgboost_{seed}"
        
        # Load the model
        model_path = os.path.join(model_dir, model_file)
        xgb_booster = xgb.Booster()
        xgb_booster.load_model(model_path)

        gain_importances = xgb_booster.get_score(importance_type='gain')

        gain_importances_list = [[key, value] for key, value in gain_importances.items()]
        # gain_importances_list = [[feature_names[int(key[1:])], value] for key, value in gain_importances.items()]

        # Sort in descending order
        gain_importances_list = sorted(gain_importances_list, key=lambda x: x[1], reverse=True)

        for feat, gain in gain_importances_list:
            gene = feature_names[int(feat[1:])]
            if gene=='CAG':
                continue
            # Find snp in lookup table
            match = snp_lookuptab[snp_lookuptab["Gene"] == gene]
            # Retrieve corresponding gene and GO term
            GOt = match['GO_term'].values[0]

            total_importance_b2 = total_importance_b2._append({
                'Feature': feat,
                'Gene': gene,
                'Score': gain,
                'GO': GOt,
                'Seed': seed
            }, ignore_index=True)

# Sort in descending order
total_importance_b2 = total_importance_b2.sort_values(by='Score', ascending=False).reset_index(drop=True)

#%%
# Group by SNP and compute mean Gain
mean_gain_df = total_importance_b2.groupby('Feature', as_index=False)['Score'].mean()
mean_gain_df = mean_gain_df.rename(columns={'Score': 'Mean_Gain'})

# Drop duplicates from original to get one Gene/GO per SNP (assumes consistency)
snp_metadata = total_importance_b2[['Feature', 'Gene', 'GO']].drop_duplicates(subset='Feature')

# Merge metadata with mean gains
snp_mean_gain_df = pd.merge(mean_gain_df, snp_metadata, on='Feature')

# sort by mean gain
snp_mean_gain_df = snp_mean_gain_df.sort_values(by='Mean_Gain', ascending=False).reset_index(drop=True)

# top 100
b2_top_genes = snp_mean_gain_df['Gene'].unique()[:100]

#%% Comparison of b1 and b2 top genes
jaccard_index(set(b1_top_genes), set(b2_top_genes))

#%% Distribution of Jaccard indices

b1_seed_sets = create_seed_sets(total_importance)
b2_seed_sets = create_seed_sets(total_importance_b2)

# Get all unique key pairs 
key_pairs = list(itertools.combinations(b1_seed_sets.keys(), 2))

jaccard1 = []
jaccard2 = []

for k1, k2 in key_pairs:
    j1 = jaccard_index(b1_seed_sets[k1], b1_seed_sets[k2])
    j2 = jaccard_index(b1_seed_sets[k1], b2_seed_sets[k2])
    jaccard1.append(j1)
    jaccard2.append(j2)

# Plot distributions
plt.figure(figsize=(8, 5))
sns.histplot(jaccard1, color='blue', label='SNP dataset', kde=True, stat='frequency', bins=10)
sns.histplot(jaccard2, color='orange', label='VAE embeddings dataset', kde=True, stat='frequency', bins=10)
plt.xlabel('Jaccard Index')
plt.ylabel('Frequency')
plt.title('Distribution of Pairwise Jaccard Indices')
plt.legend()
plt.tight_layout()
plt.savefig(f'{out_dir}b1b2_jaccard_distribution.pdf', dpi=300)
plt.show()

med_1 = np.median(jaccard1)
med_2 = np.median(jaccard2)

#%% Paired Wilcoxon test
stat, p_value = wilcoxon(jaccard1, jaccard2, )

print(f"Wilcoxon test statistic = {stat:.4f}")
print(f"P-value = {p_value:.4e}")

# %%
# Read metrics file
all_metrics = pd.read_csv(f'{out_dir}benchmark/combined_performances.txt', sep='\t')

# Check seeds of best xgboost b1 and b2 models
b1_best = all_metrics[(all_metrics['Model'] == 'XGBoost') & (all_metrics['Model Type'] == 'b1')].sort_values(by='Accuracy', ascending=False).head(1)
b2_best = all_metrics[(all_metrics['Model'] == 'XGBoost') & (all_metrics['Model Type'] == 'b2')].sort_values(by='Accuracy', ascending=False).head(1)

# %%
