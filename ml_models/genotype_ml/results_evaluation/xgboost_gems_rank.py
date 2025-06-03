#!/usr/bin/env python3

#%%### XGBoost feature contribution aggregation ###

import os
import pickle
import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt

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

total_importance = pd.DataFrame(columns=['SNP','Score','Gene','GO'])

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
                row = {'SNP':snp, 'Gain':gain, 'Gene':np.nan, 'GO':np.nan}
                booster_important_snps = booster_important_snps._append(row, ignore_index=True)
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
            existing_row = total_importance[total_importance['SNP'] == row['SNP']]
            if not existing_row.empty:
                # Add ind_score to the existing Score
                total_importance.loc[total_importance['SNP'] == row['SNP'], 'Score'] += row['Gain']
            else:
                # Add new row to total_importance
                total_importance = total_importance._append({
                    'SNP': row['SNP'],
                    'Score': row['Gain'],
                    'Gene': row['Gene'],
                    'GO': row['GO']
                }, ignore_index=True)

# Sort in descending order
total_importance = total_importance.sort_values(by='Score', ascending=False).reset_index(drop=True)

# Save total_importance table
total_importance.to_csv(f"{out_dir}b1_xgboost_total_importance.txt", sep="\t", index=False)
        
#%% Load b2 models

total_importance_b2 = pd.DataFrame(columns=['Gene','Score','GO'])

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

        gain_importances_list = [[feature_names[int(key[1:])], value] for key, value in gain_importances.items()]

        # Sort in descending order
        gain_importances_list = sorted(gain_importances_list, key=lambda x: x[1], reverse=True)

        # Assemble presenting table
        booster_important_snps = pd.DataFrame(columns=['Gene','Gain','GO'])

        for gene, gain in gain_importances_list:
            if gene=='CAG':
                row = {'Gene':gene, 'Gain':gain, 'GO':np.nan}
                booster_important_snps = booster_important_snps._append(row, ignore_index=True)
                continue
            # Find snp in lookup table
            match = snp_lookuptab[snp_lookuptab["Gene"] == gene]
            # Retrieve corresponding gene and GO term
            GOt = match['GO_term'].values[0]
            # Create new row in pd df
            row = {'Gene':gene, 'Gain':gain, 'GO':GOt}
            booster_important_snps = booster_important_snps._append(row, ignore_index=True)

        # Add contributions to total_importance table
        for _, row in booster_important_snps.iterrows():
            existing_row = total_importance_b2[total_importance_b2['Gene'] == row['Gene']]
            if not existing_row.empty:
                # Add ind_score to the existing Score
                total_importance_b2.loc[total_importance_b2['Gene'] == row['Gene'], 'Score'] += row['Gain']
            else:
                # Add new row to total_importance
                total_importance_b2 = total_importance_b2._append({
                    'Gene': row['Gene'],
                    'Score': row['Gain'],
                    'GO': row['GO']
                }, ignore_index=True)

# Sort in descending order
total_importance_b2 = total_importance_b2.sort_values(by='Score', ascending=False).reset_index(drop=True)

# Save total_importance table
total_importance_b2.to_csv(f"{out_dir}b2_xgboost_total_importance.txt", sep="\t", index=False)
