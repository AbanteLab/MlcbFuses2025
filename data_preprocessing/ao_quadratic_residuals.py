#!/usr/bin/env python3

### AO residuals after linear regression using CAG ###
#%%
# Dependencies
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle

#--------# Directories #--------#

# Change working directory
os.chdir('/pool01/projects/abante_lab/ao_prediction_enrollhd_2024/')

# Data directory
feat_dir = "features/"

#--------# Load data #--------#

# Sex and CAG vectors
sexCAG = np.load(feat_dir + 'sexCAG.npy')
cag_all = sexCAG[:,1]
cag_squared = cag_all**2
X = np.vstack([np.ones(len(cag_all)), cag_all, cag_squared]).T

# AO values
y_all = pd.read_csv(feat_dir + 'aoo.txt', sep='\t')
y_all = y_all['Onset.Age'].values

#--------# Linear Regression #--------#

coeffs, _, _, _ = np.linalg.lstsq(X, y_all, rcond=None)

# Display the regression coefficients
print("Regression coefficients (bias, cag, cag^2):", coeffs)

# Create a prediction for visualization
y_pred = X.dot(coeffs)

#--------# Save Residuals #--------#

# Compute residuals
residuals = y_all - y_pred

# Save residuals to a file
np.savetxt(feat_dir + 'AO_quadratic_residuals.txt', residuals, fmt='%.6f')

# Save model parameters
with open('ml_results/regressors/CAG2_linear_regression_params.pkl', 'wb') as f:
    pickle.dump(coeffs, f)
# %%
cag_fine = np.linspace(min(cag_all), max(cag_all), 500)
cag_squared_fine = cag_fine ** 2
X_fine = np.vstack([np.ones(len(cag_fine)), cag_fine, cag_squared_fine]).T

# Predicted values for the smoother curve
y_pred_fine = X_fine.dot(coeffs)

plt.figure(figsize=(12, 6))

# Plot original data vs predictions
plt.subplot(1, 2, 1)
plt.grid(True)
plt.scatter(cag_all, y_all, color='blue', label='Actual data')
plt.plot(cag_fine, y_pred_fine, color='red', label='Regression curve')  # Plot smooth curve
plt.xlabel('CAG Normalized')
plt.ylabel('Onset Age')
plt.title('Linear Regression: Actual vs Predicted')
plt.legend()

# Plot residuals
plt.subplot(1, 2, 2)
plt.grid(True)
plt.scatter(cag_all, residuals, color='green')
plt.axhline(y=0, color='black', linestyle='--', label='Zero residual')
plt.xlabel('CAG')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.legend()
plt.savefig("ml_results/regressors_info/AO_CAG2_regression.png")

# Plot residuals with color corresponding to true AO
plt.figure()
scatter = plt.scatter(cag_all, residuals, c=y_all, cmap='viridis')
cbar = plt.colorbar(scatter)
cbar.set_label('True AO')
plt.axhline(y=0, color='black', linestyle='--', label='Zero residual')
plt.xlabel('CAG Normalized')
plt.ylabel('Residuals')
plt.title('Residuals Colored by True AO')
plt.legend(loc = 'lower right')
plt.grid(True)
plt.savefig("ml_results/regressors_info/AO_CAG2_residuals_AOcolored.png")