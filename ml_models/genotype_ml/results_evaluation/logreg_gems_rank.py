#!/usr/bin/env python3

#%%### XGBoost feature contribution aggregation ###

import os
import pickle
import torch
import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

from logistic_regression import MultinomialLogisticRegression

os.chdir('/pool01/projects/abante_lab/ao_prediction_enrollhd_2024/')

# Data directory
data_dir = "features/"

# Path to models
model_dir = "ml_results/classification/v2/logregs/"

# Path to output directory
out_dir = "ml_results/classification/v2/" 

feat_dir = "features/"

# Read SNP lookup table
snp_lookuptab = pd.read_csv("genes/snps_gene_GO_m3.txt", sep="\t")

# Read metrics table
metrics = pd.read_csv(f"{out_dir}logisticregression_metrics_summary.txt", sep="\t")

#%% Load b1 models

# Extract feature matrix header
with open(data_dir + "subsetting/header_feature_matrix_m3_filt_0.01.txt", "r") as file:
    header = file.readline().strip().split("\t")
feature_names = header[1:]

snp_lookuptab_grouped = (
    snp_lookuptab
    .groupby("SNP")[["Gene", "GO_term"]]
    .agg(lambda x: "; ".join(sorted(set(x.dropna().astype(str)))))  # Remove duplicates
    .reset_index()
)
snp_lookup_dict = snp_lookuptab_grouped.set_index("SNP").to_dict(orient="index")
#%%
# Dictionary to store final scores
importance_dict = {}

# Process all models
for model_file in tqdm(os.listdir(model_dir), desc="Processing models"):
    if model_file.startswith("b1_logisticregression_") and model_file.endswith("_model.pkl"):
        seed = model_file.split("_")[3]
        model_name = f"b1_logisticregression_{seed}"

        with open(os.path.join(model_dir, model_file), 'rb') as file:
            model = pickle.load(file)

        weights = model.weights
        logreg_importances = torch.sum(torch.abs(weights), dim=1).cpu().numpy()

        # Collect SNP importance data
        snps_list = []

        for snp, gain in zip(feature_names, logreg_importances):
            if snp in ('CAG', 'Sex'):
                snps_list.append({'SNP': snp, 'Gain': gain, 'Gene': np.nan, 'GO': np.nan})
                continue

            if snp not in snp_lookup_dict:
                print(f"Warning: SNP {snp} not found in lookup table")
                continue

            gene, GOt = snp_lookup_dict[snp]["Gene"], snp_lookup_dict[snp]["GO_term"]
            snps_list.append({'SNP': snp, 'Gain': gain, 'Gene': gene, 'GO': GOt})

        # Convert to DataFrame once
        logreg_important_snps = pd.DataFrame(snps_list)

        # Aggregate contributions efficiently
        for _, row in logreg_important_snps.iterrows():
            snp = row['SNP']
            if snp in importance_dict:
                importance_dict[snp]['Score'] += row['Gain']
            else:
                importance_dict[snp] = {'Score': row['Gain'], 'Gene': row['Gene'], 'GO': row['GO']}

# Convert dictionary to DataFrame once
total_importance = pd.DataFrame.from_dict(importance_dict, orient='index').reset_index().rename(columns={'index': 'SNP'})

# Sort in descending order
total_importance = total_importance.sort_values(by='Score', ascending=False).reset_index(drop=True)

# Save total_importance table
total_importance.to_csv(f"{out_dir}b1_logisticregression_total_importance.txt", sep="\t", index=False)
        
# %% for b2 models

# get order of features if xgboost VAE encodings
feature_names = pd.read_csv(data_dir + "gene_tensor_order.txt", header=None)
feature_names = pd.concat((pd.DataFrame(['sex', 'CAG']), feature_names))
feature_names = feature_names[0].tolist()

# Dictionary to store final scores
importance_dict = {}

# Process all models
for model_file in tqdm(os.listdir(model_dir), desc="Processing models"):
    if model_file.startswith("b2_logisticregression_") and model_file.endswith("_model.pkl"):
        seed = model_file.split("_")[3]
        model_name = f"b2_logisticregression_{seed}"

        with open(os.path.join(model_dir, model_file), 'rb') as file:
            model = pickle.load(file)

        weights = model.weights
        logreg_importances = torch.sum(torch.abs(weights), dim=1).cpu().numpy()

        # Collect SNP importance data
        feats_list = []

        for feat, gain in zip(feature_names, logreg_importances):
            if feat in ('CAG', 'sex'):
                feats_list.append({'SNP': feat, 'Gain': gain, 'Gene': np.nan, 'GO': np.nan})
                continue

            if feat not in snp_lookuptab['Gene'].values:
                print(f"Warning: gene {feat} not found in lookup table")
                continue

            GOt = snp_lookuptab[snp_lookuptab["Gene"]==feat]["GO_term"].iloc[0]
            feats_list.append({'Gene': feat, 'Gain': gain, 'GO': GOt})

        # Convert to DataFrame once
        logreg_important_snps = pd.DataFrame(feats_list)

        # Aggregate contributions efficiently
        for _, row in logreg_important_snps.iterrows():
            gene = row['Gene']
            if gene in importance_dict:
                importance_dict[gene]['Score'] += row['Gain']
            else:
                importance_dict[gene] = {'Score': row['Gain'], 'GO': row['GO']}

# Convert dictionary to DataFrame once
total_importance = pd.DataFrame.from_dict(importance_dict, orient='index').reset_index().rename(columns={'index': 'Gene'})

# Sort in descending order
total_importance = total_importance.sort_values(by='Score', ascending=False).reset_index(drop=True)

# Save total_importance table
total_importance.to_csv(f"{out_dir}b2_logisticregression_total_importance.txt", sep="\t", index=False)
        
# %%
