#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 09:29:05 2024

@author: jaanajurvansuu
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

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

# Manually define boundary colors
boundary_color_dict = {
    'WWTP6': '#A3CBE2',
    'WWTP2': '#A3A3E0',
    'WWTP3': '#FFEB99',
    'WWTP4': '#FFC299',
    'WWTP1': '#FF99FF',
    'WWTP5': '#D699D6'
}

# Select best model features
selected_features_indices = [20, 35, 43, 60, 101, 105, 108, 120, 140, 448]
X_selected = X.iloc[:, selected_features_indices].copy()

# Add the selected features and target to a new DataFrame
X_selected['SAMPLE'] = y.values 

# Decision Boundary Plots
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected.drop('SAMPLE', axis=1))

# PCA transformation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Get the explained variance
explained_variance = pca.explained_variance_ratio_

# Fit GaussianNB model
model = GaussianNB()
model.fit(X_pca, y_encoded)

# Meshgrid for plotting
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Calculate probabilities
probs = model.predict_proba(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape + (len(class_labels),))

# Plot probability contours with correct colors
plt.figure(figsize=(10, 6))

# Manually assign colors for each class boundary
for cls, class_label in enumerate(class_labels):
    lighter_color = boundary_color_dict[class_label]
    plt.contourf(xx, yy, probs[:, :, cls], levels=np.linspace(0.1, 1, 11), alpha=0.5, colors=[lighter_color])

# Plot each class scatter points
for cls in np.unique(y_encoded):
    class_label = class_labels[cls]
    plt.scatter(X_pca[y_encoded == cls, 0], X_pca[y_encoded == cls, 1], label=class_label, edgecolor='k', s=20, color=color_dict[class_label])

plt.xlabel(f'PCA Component 1 ({explained_variance[0]*100:.2f}% variance)')
plt.ylabel(f'PCA Component 2 ({explained_variance[1]*100:.2f}% variance)')
#plt.title('GaussianNB Probability Contours')
plt.legend()
plt.tight_layout()
plt.savefig('probability_contours_plot.png')
plt.close()
