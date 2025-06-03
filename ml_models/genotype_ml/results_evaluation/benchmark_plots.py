#!/usr/bin/env python3

### Plots for benchmarking ###
#%%
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# Load data
results_dir = "/pool01/projects/abante_lab/ao_prediction_enrollhd_2024/ml_results/classification/v2/benchmark/"
xb_file_path = "/pool01/projects/abante_lab/ao_prediction_enrollhd_2024/ml_results/classification/v2/xgboost_metrics_summary.txt"
lr_file_path = "/pool01/projects/abante_lab/ao_prediction_enrollhd_2024/ml_results/classification/v2/logisticregression_metrics_summary.txt"
gnns_file_path = "/pool01/projects/abante_lab/ao_prediction_enrollhd_2024/gnn/classification/gnn_metrics_summary.txt"
lr = pd.read_csv(lr_file_path, sep="\t")
xb = pd.read_csv(xb_file_path, sep="\t")
gnns = pd.read_csv(gnns_file_path, sep="\t")

# Identify model types
lr["Model Type"] = lr["Model"].apply(lambda x: "b1" if "b1" in x else 
                                     ("b2" if "b2" in x else "Other"))
xb["Model Type"] = xb["Model"].apply(lambda x: "b1" if "b1" in x else 
                                     ("b2" if "b2" in x else "Other"))

# Record model seed
lr["Seed"] = lr['Model'].str.extract(r'_(\d+)$').astype(int)
xb["Seed"] = xb['Model'].str.extract(r'_(\d+)$').astype(int)

# Filter only b1 and b2 models
lr = lr[lr["Model Type"].isin(["b1", "b2"])]
xb = xb[xb["Model Type"].isin(["b1", "b2"])]

# Convert columns to numeric
lr["Accuracy"] = pd.to_numeric(lr["Accuracy"])
xb["Accuracy"] = pd.to_numeric(xb["Accuracy"])
lr["Train Balanced Accuracy"] = pd.to_numeric(lr["Train Balanced Accuracy"])
xb["Train Balanced Accuracy"] = pd.to_numeric(xb["Train Balanced Accuracy"])
lr["LR"] = pd.to_numeric(lr["LR"])
lr["Lambda_L1"] = pd.to_numeric(lr["Lambda_L1"])
xb["Depth"] = pd.to_numeric(xb["Depth"])
xb["LR"] = pd.to_numeric(xb["Learning Rate"])
xb = xb.drop(columns=["Learning Rate"])

# Convert additional columns to numeric
columns_to_convert = ["Precision", "Recall", "F1-Score", "Support"]
for col in columns_to_convert:
    lr[col] = pd.to_numeric(lr[col])
    xb[col] = pd.to_numeric(xb[col])

# Combine datasets
lr["Model"] = "Logistic Regression"
xb["Model"] = "XGBoost"
gnns["Model"] = gnns["Model_Seed"].apply(lambda x: "GNNConvDropoutPool" if "GNNConvDropoutPool" in x else 
                                     ("EdgeGAT" if "EdgeGAT" in x else "Other"))
gnns["Seed"] = gnns['Model_Seed'].str.extract(r'_(\d+)$').astype(int)
gnns = gnns.drop(columns = "Model_Seed")
gnns["Model Type"] = 'b2'
combined = pd.concat([lr, xb, gnns], ignore_index=True)
combined.to_csv(f"{results_dir}combined_performances.txt", sep= "\t", index = False)

best_models = combined.loc[combined.groupby('Model')['Accuracy'].idxmax()]

#%% benchmark 1 and 2 separately
# Plot 1: Benchmark 1
plt.figure(figsize=(8, 6))
sns.violinplot(x="Model", y="Accuracy", data=combined[combined["Model Type"] == "b1"])
plt.title("SNPs genotype features benchmark")
plt.ylabel("Testing Balanced Accuracy")
plt.ylim(0.18, 0.3)
plt.savefig(f"{results_dir}b1_accuracy_comparison.png")
plt.savefig(f"ml_results/classification/v2/paperfigs/b1_accuracy_comparison.pdf", dpi = 300)
plt.show()

# Plot 2: Benchmark 2
plt.figure(figsize=(8, 6))
sns.violinplot(x="Model", y="Accuracy", data=combined[combined["Model Type"] == "b2"])
plt.title("Second benchmark (SNPs genotype + VAE encodings features)")
plt.savefig(f"{results_dir}b2_accuracy_comparison.png")
plt.show()

#%% Wilcoxon test of b1 models
subset = combined[combined["Model Type"] == "b1"]

# Make sure the data is aligned by index or sample ID
paired_data = subset.pivot(index='Seed', columns='Model', values='Accuracy')  # adjust if needed

stat, p = wilcoxon(paired_data["Logistic Regression"], paired_data["XGBoost"])
print(f"Wilcoxon test statistic: {stat}, p-value: {p}")

#%% XGBoosts comparison

# Violin plot comparing accuracies of b1 and b2 models
plt.figure(figsize=(8, 6))
sns.violinplot(x="Model Type", y="Accuracy", data=xb)
plt.title("Accuracy Comparison Between b1 and b2 Models")
plt.savefig(f"{results_dir}xgboost_accuracy_comparison.png")
plt.show()

# box plot for each hyperparameter
for hyperparam in ["Depth", "Learning Rate"]:
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x="Model Type", y="Accuracy", hue=hyperparam, data=xb, palette="muted")

    # Get unique x-ticks and hue categories
    model_types = xb["Model Type"].unique()
    hyper_values = xb[hyperparam].unique()
    
    # Offsets for proper positioning of text within each x-tick group
    offsets = np.linspace(-0.27, 0.27, len(hyper_values))

    # Get lower y-bound to place text near the bottom
    y_min = xb["Accuracy"].min() - (xb["Accuracy"].max() - xb["Accuracy"].min()) * 0.05  # Slightly below min value

    # Compute and annotate counts
    grouped_counts = xb.groupby(["Model Type", hyperparam]).size().reset_index(name="Count")

    for i, row in grouped_counts.iterrows():
        model_type = row["Model Type"]
        hyper_value = row[hyperparam]
        count = row["Count"]

        # Compute x position with hue offset
        x_position = list(model_types).index(model_type)
        hue_index = list(hyper_values).index(hyper_value)
        x_adjusted = x_position + offsets[hue_index]

        # Annotate count near bottom of y-axis
        ax.text(x_adjusted, y_min, f'n={count}', ha='center', va='bottom', fontsize=10, color='black')

    plt.title(f"Accuracy by {hyperparam} for b1 and b2 Models")
    plt.legend(title=hyperparam)
    plt.ylim(y_min, xb["Accuracy"].max() * 1.05)  # Adjust y-axis to fit text
    plt.savefig(f"{results_dir}xgboost_accuracy_by_{hyperparam.lower().replace(' ', '_')}.pdf")
    plt.show()

# Same for training accuracy

# box plot for each hyperparameter
for hyperparam in ["Depth", "Learning Rate"]:
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x="Model Type", y="Train Balanced Accuracy", hue=hyperparam, data=xb, palette="muted")

    # Get unique x-ticks and hue categories
    model_types = xb["Model Type"].unique()
    hyper_values = xb[hyperparam].unique()
    
    # Offsets for proper positioning of text within each x-tick group
    offsets = np.linspace(-0.27, 0.27, len(hyper_values))

    # Get lower y-bound to place text near the bottom
    y_min = xb["Train Balanced Accuracy"].min() - (xb["Train Balanced Accuracy"].max() - xb["Train Balanced Accuracy"].min()) * 0.05  # Slightly below min value

    # Compute and annotate counts
    grouped_counts = xb.groupby(["Model Type", hyperparam]).size().reset_index(name="Count")

    for i, row in grouped_counts.iterrows():
        model_type = row["Model Type"]
        hyper_value = row[hyperparam]
        count = row["Count"]

        # Compute x position with hue offset
        x_position = list(model_types).index(model_type)
        hue_index = list(hyper_values).index(hyper_value)
        x_adjusted = x_position + offsets[hue_index]

        # Annotate count near bottom of y-axis
        ax.text(x_adjusted, y_min, f'n={count}', ha='center', va='bottom', fontsize=10, color='black')

    plt.title(f"Train Accuracy by {hyperparam} for b1 and b2 Models")
    plt.legend(title=hyperparam)
    plt.ylim(y_min, xb["Train Balanced Accuracy"].max() * 1.05)  # Adjust y-axis to fit text
    plt.savefig(f"{results_dir}xgboost_train_accuracy_by_{hyperparam.lower().replace(' ', '_')}.pdf")
    plt.show()
# %% Logistic regression comparison

# Violin plot comparing accuracies of b1 and b2 models
plt.figure(figsize=(8, 6))
sns.violinplot(x="Model Type", y="Accuracy", data=lr)
plt.title(f"Accuracy Comparison Between b1 and b2 Models (Logistic Regression)")
plt.savefig(f"{results_dir}logisticregression_accuracy_comparison.png")
plt.show()

# Plot 2: box plot for each hyperparameter within each model type

plt.figure(figsize=(10, 6))
sns.boxplot(x="Model Type", y="Accuracy", hue="Lambda_L1", data=lr, palette="muted")
plt.title("Accuracy by Lambda_L1 for b1 and b2 Models (logisticregression)")
plt.legend(title="Lambda_L1")
plt.savefig(f"{results_dir}logisticregression_accuracy_by_Lambda_L1.png")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x="Model Type", y="Train Balanced Accuracy", hue="Lambda_L1", data=lr, palette="muted")
plt.title("Train Balanced Accuracy by Lambda_L1 for b1 and b2 Models (logisticregression)")
plt.legend(title="Lambda_L1")
plt.savefig(f"{results_dir}logisticregression_train_accuracy_by_Lambda_L1.png")
plt.show()
# %%
# Unique violin plot with specified order and colors
plt.figure(figsize=(12, 8))
order = ["Logistic Regression", "XGBoost"]
combined["Model Group"] = combined["Model"].apply(lambda x: "Logistic Regression" if x == "Logistic Regression" else "XGBoost")
sns.violinplot(x="Model Group", y="Accuracy", hue="Model Type", data=combined, order=order, palette={"b1": "palevioletred", "b2": "lightskyblue"})
handles = [plt.Line2D([0], [0], color="palevioletred", lw=4, label="SNPs dataset"),
           plt.Line2D([0], [0], color="lightskyblue", lw=4, label="SNPs and VAE embeddings dataset")]
plt.legend(handles=handles, loc="upper left", fontsize=14, title_fontsize=14, frameon=True, framealpha=1, shadow=False, borderpad=1, edgecolor='black')
plt.xlabel('')
plt.ylabel('Balanced Test Accuracy', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig(f"{results_dir}unique_violin_plot.pdf", dpi=300)
plt.show()

#%% b1 and b2 comparison
# Compute median accuracy for each model
median_accuracies = combined.groupby("Model")["Accuracy"].median()

# Sort models by ascending median accuracy
order = median_accuracies.sort_values().index.tolist()

plt.figure(figsize=(12, 6))

# Assign a new column just for plotting purposes
def model_type_for_plot(row):
    if row["Model"] in ["Logistic Regression", "XGBoost"]:
        return row["Model Type"]
    else:
        return "b2"  # Or any single value; they won't be split

combined["Model Type For Plot"] = combined.apply(model_type_for_plot, axis=1)

# Plot everything together with hue
sns.violinplot(
    x="Model",
    y="Accuracy",
    hue="Model Type For Plot",
    data=combined,
    order=order,
    palette={"b1": "palevioletred", "b2": "lightskyblue"},
    split=False
)

# Custom legend
handles = [
    plt.Line2D([0], [0], color="palevioletred", lw=4, label="SNPs dataset"),
    plt.Line2D([0], [0], color="lightskyblue", lw=4, label="SNPs and VAE embeddings dataset")
]
plt.legend(handles=handles, loc="lower right", fontsize=14, title_fontsize=14, frameon=True, framealpha=1, shadow=False, borderpad=1, edgecolor='black')

# Labels and formatting
plt.xlabel('')
plt.ylabel('Balanced Test Accuracy', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.savefig(f"{results_dir}b1b2_combined_violin_plot.pdf", dpi=300)

plt.show()

#%% Violin plot for b2 test accuracy
plt.figure(figsize=(12, ))
order = ["Logistic Regression", "XGBoost", "GNNConvDropoutPool", "EdgeGAT"]
combined["Model Group"] = combined["Model"].apply(lambda x: "Logistic Regression" if x == "Logistic Regression" else 
                                                  ("XGBoost" if x == "XGBoost" else 
                                                   ("GNNConvDropoutPool" if x == "GNNConvDropoutPool" else "EdgeGAT")))
combined = combined[combined["Model Type"] == "b2"]
sns.violinplot(x="Model Group", y="Accuracy", data=combined, order=order, color="lightskyblue")
plt.xlabel('')
plt.ylabel('Balanced Test Accuracy', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig(f"{results_dir}benchmark2_unique_violin_plot.pdf", dpi=300)
plt.show()
#%% # Violin plot for b2 train accuracy
plt.figure(figsize=(12, 8))
order = ["Logistic Regression", "XGBoost", "GNNConvDropoutPool", "EdgeGAT"]
combined["Model Group"] = combined["Model"].apply(lambda x: "Logistic Regression" if x == "Logistic Regression" else 
                                                  ("XGBoost" if x == "XGBoost" else 
                                                   ("GNNConvDropoutPool" if x == "GNNConvDropoutPool" else "EdgeGAT")))
combined = combined[combined["Model Type"] == "b2"]
sns.violinplot(x="Model Group", y="Train Balanced Accuracy", data=combined, order=order, color="lightskyblue")
plt.xlabel('')
plt.ylabel('Train Balanced Accuracy', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig(f"{results_dir}benchmark2_train_acc.pdf", dpi=300)
plt.show()

#%% Other metrics benchmark
metrics = ["Precision", "Recall", "F1-Score"]
order = ["Logistic Regression", "XGBoost"]
combined["Model Type Combined"] = combined["Model"] + " " + combined["Model Type"]
combined["Model Group"] = combined["Model"].apply(lambda x: "Logistic Regression" if x == "Logistic Regression" else "XGBoost")

for metric in metrics:
    plt.figure(figsize=(12, 8))
    sns.violinplot(x="Model Group", y=metric, hue="Model Type", data=combined, order=order, palette={"b1": "palevioletred", "b2": "lightskyblue"})
    handles = [plt.Line2D([0], [0], color="palevioletred", lw=4, label="SNPs dataset"),
               plt.Line2D([0], [0], color="lightskyblue", lw=4, label="SNPs and VAE embeddings dataset")]
    plt.legend(handles=handles, loc="best", fontsize=14, title_fontsize=14, frameon=True, framealpha=1, shadow=False, borderpad=1, edgecolor='black')
    plt.xlabel('')
    plt.ylabel(metric, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(f"{metric} Comparison Between Models")
    plt.savefig(f"{results_dir}unique_violin_plot_{metric.lower().replace(' ', '_')}.png", dpi=300)
    plt.show()
# %% Recall per class
plt.figure(figsize=(16, 8))
order = [0, 1, 2, 3, 4]
classes = [0, 1, 2, 3, 4]

# Melt the dataset to have a single "Recall" column and a "Class" column
recall_columns = [f"Recall_{cls}" for cls in classes]
if all(col in combined.columns for col in recall_columns):
    melted_combined = combined.melt(id_vars=["Model", "Model Type"], 
                                    value_vars=recall_columns, 
                                    var_name="Class", 
                                    value_name="Class recall")
    melted_combined["Class"] = melted_combined["Class"].str.extract(r'Recall_(\d)').astype(int)
    melted_combined["Model Group"] = melted_combined["Model"].apply(lambda x: "Logistic Regression" if x == "Logistic Regression" else 
                                                                    ("XGBoost" if x == "XGBoost" else 
                                                                     ("GNNConvDropoutPool" if x == "GNNConvDropoutPool" else "EdgeGAT")))
    melted_combined = melted_combined[melted_combined["Model Type"] == "b2"]

    # Create a violin plot with models separated for each class
    sns.violinplot(x="Class", y="Class recall", hue="Model Group", data=melted_combined, 
                   order=order, palette="muted", dodge=True, split=False)
    
    # Adjust spacing between classes
    for i, class_label in enumerate(order):
        plt.axvline(i + 0.5, color='gray', linestyle='--', alpha=0.5)

    # Add horizontal grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.xlabel('Class')
    plt.ylabel('Recall')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(title="Model", loc="upper right")
    plt.savefig(f"{results_dir}benchmark2_recall_by_model_group_separated.pdf", dpi=300)
    plt.show()
else:
    print("Error: One or more recall columns are missing in the dataset.")

#%% Statistical tests for performance metrics using ANOVA

# Function to perform ANOVA and return results
def perform_anova(groups, metric):
    stat, p_value = f_oneway(*[group[metric] for group in groups])
    return {
        "Metric": metric,
        "Statistic": stat,
        "P-value": p_value
    }

# Define model groups
logistic_b1 = combined[(combined["Model"] == "Logistic Regression") & (combined["Model Type"] == "b1")]
logistic_b2 = combined[(combined["Model"] == "Logistic Regression") & (combined["Model Type"] == "b2")]
xgboost_b1 = combined[(combined["Model"] == "XGBoost") & (combined["Model Type"] == "b1")]
xgboost_b2 = combined[(combined["Model"] == "XGBoost") & (combined["Model Type"] == "b2")]

# Metrics to compare
metrics = ["Accuracy", "Precision", "F1-Score"]

# Perform ANOVA and collect results
results = []
groups = [logistic_b1, logistic_b2, xgboost_b1, xgboost_b2]
for metric in metrics:
    results.append(perform_anova(groups, metric))

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results to a CSV file
results_df.to_csv(f"{results_dir}anova_results.csv", index=False)

# Print results
print(results_df)
# %%
