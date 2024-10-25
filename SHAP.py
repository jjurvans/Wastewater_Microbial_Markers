#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:50:13 2024

@author: jaanajurvansuu
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.naive_bayes import GaussianNB
import shap

# Set the default font weight to 'normal' for all fonts
plt.rcParams.update({'font.weight': 'normal'})
plt.rcParams.update({
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 14
})

# Load data
df = pd.read_csv('CodeEffsDataframe.csv')

# Separate features and target
X = df.drop('SAMPLE', axis=1)
y = df['SAMPLE']

# Encode the target variable if it's categorical
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_labels = le.classes_

# Color dictionary for the classes
color_dict = {
    'WWTP6': '#007BA7',
    'WWTP2': '#0000CD',
    'WWTP3': '#FFD300',
    'WWTP4': '#FF5F00',
    'WWTP1': '#FF00FF',
    'WWTP5': '#800080'
}

# Select best model features
selected_features_indices = [20, 35, 43, 60, 101, 105, 108, 120, 140, 448]
selected_feature_names = X.columns[selected_features_indices]
X_selected = X.iloc[:, selected_features_indices].copy()

# Add the selected features and target to a new DataFrame
X_selected['SAMPLE'] = y.values 

# SHAP Summary Plots using KernelExplainer
model = GaussianNB()
model.fit(X_selected.drop('SAMPLE', axis=1), y_encoded)

# Use SHAP KernelExplainer to explain the model
background_sample_size = min(100, X_selected.shape[0])
explainer = shap.KernelExplainer(model.predict_proba, X_selected.drop('SAMPLE', axis=1).sample(background_sample_size, random_state=42))
shap_values = explainer.shap_values(X_selected.drop('SAMPLE', axis=1))

# Extract SHAP values and calculate mean absolute SHAP values
mean_abs_shap_values = []
for class_idx, class_label in enumerate(class_labels):
    class_shap_values = np.abs(shap_values[class_idx])
    mean_abs_shap_values.append(np.mean(class_shap_values, axis=0))

# Sort feature order from 0 to 10
feature_order = np.arange(len(selected_feature_names))

# Plot SHAP values manually with consistent feature order
fig, axes = plt.subplots(3, 2, figsize=(20, 30))
axes = axes.flatten()

for i, class_label in enumerate(class_labels):
    ax = axes[i]
    shap_df = pd.DataFrame({
        'Feature': selected_feature_names[feature_order],
        'Mean(|SHAP value|)': mean_abs_shap_values[i][feature_order]
    })
    sns.barplot(x='Mean(|SHAP value|)', y='Feature', data=shap_df, ax=ax, color=color_dict[class_label])
    ax.set_title([class_label])
    ax.set_xlabel('Mean(|SHAP value|)')
    ax.set_ylabel('')
    
# Adjust layout to make space for the titles and x-labels
plt.tight_layout(rect=[0, 0.05, 1, 0.97])

# Save the final figure
plt.savefig('shap_summary_plots.png')
plt.show()
