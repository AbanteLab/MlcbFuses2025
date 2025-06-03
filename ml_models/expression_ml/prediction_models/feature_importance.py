#!/usr/bin/env python3

#%%### XGBoost feature contribution aggregation ###

import os
import pickle
import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

lookuptab = pl.read_csv("/pool01/databases/biomart/hsapiens_ensembl_table.tsv", separator="\t")

os.chdir('/pool01/projects/abante_lab/')

# Data directory
data_dir = "ao_prediction_enrollhd_2024/features/"
borzoi_dir = "genomic_llms/borzoi/proc_results/expression_vectors/"

putamen_path = borzoi_dir + "putamen/"
caudate_path = borzoi_dir + "caudate/"

# Results directory
results_dir = "genomic_llms/multimodal_prediction/"

#--------# Load feature names #--------#

# Load expression vectors
putamen_files = [f for f in os.listdir(putamen_path) if f.endswith(".txt.gz")]
expression_vectors_putamen = [pl.read_csv(os.path.join(putamen_path, f), separator="\t") for f in putamen_files]

# Keep sample column only from the first dataframe
base = expression_vectors_putamen[0]
others = [df.drop("sample") for df in expression_vectors_putamen[1:]]

# Concatenate all horizontally
expression_vectors_putamen = pl.concat([base] + others, how="horizontal")

caudate_files = [f for f in os.listdir(caudate_path) if f.endswith(".txt.gz")]
expression_vectors_caudate = [pl.read_csv(os.path.join(caudate_path, f), separator="\t") for f in caudate_files]

# Keep sample column only from the first dataframe
base = expression_vectors_caudate[0]
others = [df.drop("sample") for df in expression_vectors_caudate[1:]]

# Concatenate all horizontally
expression_vectors_caudate = pl.concat([base] + others, how="horizontal")

# List of header lists (first putamen, second caudate)
headers_ensmbl = [expression_vectors_putamen.drop("sample").columns, expression_vectors_caudate.drop("sample").columns, expression_vectors_putamen.drop("sample").columns + expression_vectors_caudate.drop("sample").columns]

# Change gene name format to symbol, leve ensembl_gene_id in null symbol cases
headers = []
for header_ensmbl in headers_ensmbl:
    header = (
        pl.DataFrame({"ensembl_gene_id": header_ensmbl})
        .join(lookuptab, on="ensembl_gene_id", how="left")
        .with_columns(
            pl.coalesce(["external_gene_name", "ensembl_gene_id"]).alias("header")
        )
        .select("header")
        .to_series()
        .to_list()
    )
    headers.append(header)

# Add tissue to gene name
putamen_header = [col + "_putamen" for col in headers[0]]
caudate_header = [col + "_caudate" for col in headers[1]]
header_tissue_nosexcag =  putamen_header + caudate_header

# Add sex and CAG to header
header_tissue = ["Sex_any", "CAG_any"] + header_tissue_nosexcag

# Header with no tissue information
header = ["Sex", "CAG"] + headers[2]

# Read metrics table
metrics = pd.read_csv(f"{results_dir}xgboost_metrics_summary.txt", sep="\t")

# Create column for dataset
metrics['Dataset'] = metrics['Model'].apply(lambda x: x.split('_')[0])

metrics[metrics['Dataset'] == 'expression'].loc[metrics[metrics['Dataset'] == 'expression']['Accuracy'].idxmax()]

#%% Load models only expression

total_importance = pd.DataFrame(columns=['Gene','Tissue','Score'])

# Loop through all models in the model_dir that start with "b1"
for model_file in os.listdir(results_dir):
    if model_file.startswith("expression_ES_xgboost_") and model_file.endswith("_model.json"):
        
        # model_file = "expression_ES_xgboost_15_model.json"
        
        # Extract model name from the filename
        seed = model_file.split("_")[3]
        model_name = f"expression_ES_xgboost_{seed}"

        # acc = metrics[metrics['Model'] == model_name]['Accuracy'].values[0]
        
        # Load the model
        model_path = os.path.join(results_dir, model_file)
        xgb_booster = xgb.Booster()
        xgb_booster.load_model(model_path)

        gain_importances = xgb_booster.get_score(importance_type='gain')

        gain_importances_list = [[header_tissue[int(key[1:])], value] for key, value in gain_importances.items()]

        # Sort in descending order
        gain_importances_list = sorted(gain_importances_list, key=lambda x: x[1], reverse=True)

        # Create a DataFrame from the list
        booster_important_snps = pd.DataFrame(gain_importances_list, columns=['Gene', 'Gain'])

        # Add contributions to total_importance table
        for _, row in booster_important_snps.iterrows():
            existing_row = total_importance[(total_importance['Gene'] == row['Gene'].split("_")[0]) & (total_importance['Tissue'] == row['Gene'].split("_")[1])]
            if not existing_row.empty:
                # Add ind_score to the existing Score
                total_importance.loc[(total_importance['Gene'] == row['Gene'].split("_")[0]) & (total_importance['Tissue'] == row['Gene'].split("_")[1]), 'Score'] += row['Gain']
            else:
                # Add new row to total_importance
                total_importance = total_importance._append({
                    'Gene': row['Gene'].split("_")[0],
                    'Tissue': row['Gene'].split("_")[1],
                    'Score': row['Gain']
                }, ignore_index=True)

# Sort in descending order
total_importance = total_importance.sort_values(by='Score', ascending=False).reset_index(drop=True)

# Print white spaced list of genes
print(*total_importance['Gene'].head(100).to_list(), sep=' ')

# Save total_importance table
total_importance.to_csv(f"{results_dir}expression_xgboost_total_importance.txt", sep="\t", index=False)

#%% Look at enhancer loci of important expression vectors in the models

#  Read all prediction files and concatenate them
def predictions(tissue):
    pred_path = f"genomic_llms/borzoi/proc_results/weighted_logSED/{tissue}/"
    pred_files = [f for f in os.listdir(pred_path) if f.endswith(".tsv.gz")]
    predictions = [pl.read_csv(os.path.join(pred_path, f), separator="\t") for f in pred_files]
    
    # Stack all predictions vertically
    predictions = pl.concat(predictions, how="vertical")

    return predictions

caudate_predictions = predictions("caudate")
putamen_predictions = predictions("putamen")

lookuptab_renamed = lookuptab.rename({'ensembl_gene_id': 'gene'})

# Change gene_id to ensembl_gene_id
caudate_predictions = caudate_predictions.join(
    lookuptab_renamed, on='gene', how='left'
).with_columns(
    pl.col('external_gene_name').fill_null(pl.col('gene')).alias('gene_symbol')
)

putamen_predictions = putamen_predictions.join(
    lookuptab_renamed, on='gene', how='left'
).with_columns(
    pl.col('external_gene_name').fill_null(pl.col('gene')).alias('gene_symbol')
)

# Concatenate predictions
predictions = pl.concat([caudate_predictions, putamen_predictions])

# Filter predictions for top 100 important features
for gene in total_importance['Gene'].to_list()[1:101]:
    pred_snps = predictions.filter(pl.col('gene_symbol')==gene)['snp'].unique().sort().to_list()
    # Save pred_snps to file
    with open(f"genomic_llms/multimodal_prediction/features/featurespred_snps_{gene}.txt", "w") as file:
        file.write("\n".join(pred_snps))

# Save gene symbols to file
with open(f"genomic_llms/multimodal_prediction/features/featurespred_all_genes.txt", "w") as file:
    file.write("\n".join(total_importance['Gene'].to_list()[1:101]))

#%% Multimodal predictions

metrics[metrics['Dataset'] == 'multimodal'].loc[metrics[metrics['Dataset'] == 'multimodal']['Accuracy'].idxmax()]

# For multimodal xgboosts, concatenate to genotype header
with open(data_dir + "subsetting/header_feature_matrix_m3_filt_0.01.txt", "r") as file:
    header_genotype = file.readline().strip().split("\t")
header_genotype = header_genotype[1:]

header_genotype = [col + "_any" for col in header_genotype]

header = header_genotype + header_tissue_nosexcag

multimodal_total_importance = pd.DataFrame(columns=['Gene','Tissue','Score'])

# Loop through all models in the model_dir that start with "b1"
for model_file in os.listdir(results_dir):
    if model_file.startswith("multimodal_ES_xgboost_") and model_file.endswith("_model.json"):
        
        # model_file = "multimodal_ES_xgboost_27_model.json"
        
        # Extract model name from the filename
        seed = model_file.split("_")[3]
        model_name = f"multimodal_ES_xgboost_{seed}"

        # acc = metrics[metrics['Model'] == model_name]['Accuracy'].values[0]
        
        # Load the model
        model_path = os.path.join(results_dir, model_file)
        xgb_booster = xgb.Booster()
        xgb_booster.load_model(model_path)

        gain_importances = xgb_booster.get_score(importance_type='gain')

        gain_importances_list = [[header[int(key[1:])], value] for key, value in gain_importances.items()]

        # Sort in descending order
        gain_importances_list = sorted(gain_importances_list, key=lambda x: x[1], reverse=True)

        # Create a DataFrame from the list
        booster_important_snps = pd.DataFrame(gain_importances_list, columns=['Gene', 'Gain'])

        # Add contributions to total_importance table
        for _, row in booster_important_snps.iterrows():
            existing_row = multimodal_total_importance[(multimodal_total_importance['Gene'] == row['Gene'].split("_")[0]) & (multimodal_total_importance['Tissue'] == row['Gene'].split("_")[1])]
            if not existing_row.empty:
                # Add ind_score to the existing Score
                multimodal_total_importance.loc[(multimodal_total_importance['Gene'] == row['Gene'].split("_")[0]) & (multimodal_total_importance['Tissue'] == row['Gene'].split("_")[1]), 'Score'] += row['Gain']
            else:
                # Add new row to total_importance
                multimodal_total_importance = multimodal_total_importance._append({
                    'Gene': row['Gene'].split("_")[0],
                    'Tissue': row['Gene'].split("_")[1],
                    'Score': row['Gain']
                }, ignore_index=True)

# Sort in descending order
multimodal_total_importance = multimodal_total_importance.sort_values(by='Score', ascending=False).reset_index(drop=True)

non_snp_genes = multimodal_total_importance[~multimodal_total_importance['Gene'].str.startswith('rs')]
top_importances = multimodal_total_importance.head(100)
non_snp_top = top_importances[~top_importances['Gene'].str.startswith('rs')]
print('Fraction of top100 features used in models corresponding to expression features:', len(non_snp_top)/100)

mean_tissue_total_importance = multimodal_total_importance.groupby('Gene', as_index=False)['Score'].mean().sort_values(by='Score', ascending=False).reset_index(drop=True)

# Look for Gene that doesn't start with rs
top_importances = mean_tissue_total_importance.head(100)
non_snp_top = top_importances[~top_importances['Gene'].str.startswith('rs')]
print('Fraction of top100 features used in models corresponding to expression features, averaging by gene:', len(non_snp_top)/100)

# Print white spaced list of genes
# print(*total_importance['Gene'].to_list(), sep=' ')

# print(*non_snp_top['Gene'].to_list()[1:101], sep=' ')

# Save total_importance table
multimodal_total_importance.to_csv(f"{results_dir}multimodal_xgboost_total_importance.txt", sep="\t", index=False)