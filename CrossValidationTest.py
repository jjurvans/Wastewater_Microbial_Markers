#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 12:57:49 2024

@author: jaanajurvansuu
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, permutation_test_score, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportion_confint

# Set a global random seed
np.random.seed(42)

# Load data
df = pd.read_csv('CodeEffsDataframe.csv')

# Separate features and target
X = df.drop('SAMPLE', axis=1)
y = df['SAMPLE']

# Encode target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Manually set selected feature indices based on the optimization script
selected_features_indices = [20, 35, 43, 60, 101, 105, 108, 120, 140, 448]
X_selected = X.iloc[:, selected_features_indices]

# Print out the features by name
selected_feature_names = X.columns[selected_features_indices]
print(f"Selected feature names: {selected_feature_names}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Number of samples in the test set
n_samples_test = len(X_test)
print(f"Number of samples in the test set: {n_samples_test}")

# Print class distribution before SMOTE
print("Class distribution before SMOTE:")
print(pd.Series(y_train).value_counts())

# Function to create sampling strategy dictionary and set k_neighbors dynamically
def create_smote_instance(y_train, ratio):
    class_counts = np.bincount(y_train)
    majority_class = np.max(class_counts)
    sampling_strategy = {class_idx: max(class_counts[class_idx], int(majority_class * ratio)) for class_idx in range(len(class_counts))}
    
    # Determine the number of samples in the smallest class in y_train
    minority_class_count = np.min(class_counts[class_counts > 0])
    k_neighbors = max(1, min(5, minority_class_count - 1))
    
    return SMOTE(random_state=42, sampling_strategy=sampling_strategy, k_neighbors=k_neighbors)

# Apply the improved SMOTE instance
smote = create_smote_instance(y_train, 0.4)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Print class distribution after SMOTE
print("Class distribution after SMOTE:")
print(pd.Series(y_train_resampled).value_counts())

# Convert X_train_resampled and y_train_resampled to numpy arrays for indexing
X_train_resampled = X_train_resampled.to_numpy()
y_train_resampled = np.array(y_train_resampled)
X_test = X_test.to_numpy()  # Convert X_test to numpy array

# Train GaussianNB model and calculate default accuracy on resampled data
model = GaussianNB()
model.fit(X_train_resampled, y_train_resampled)
y_pred_resampled = model.predict(X_test)
default_accuracy_resampled = accuracy_score(y_test, y_pred_resampled)
print(f"Default Accuracy on Resampled Data: {default_accuracy_resampled}")

# Confidence interval for default accuracy on resampled data
n_correct_default_resampled = default_accuracy_resampled * n_samples_test
ci_default_resampled = proportion_confint(count=n_correct_default_resampled, nobs=n_samples_test, alpha=0.05, method='normal')
print(f"95% confidence interval for default accuracy on resampled data: {ci_default_resampled}")

# Train GaussianNB model and calculate default accuracy on original data
model_original = GaussianNB()
model_original.fit(X_train, y_train)
y_pred_original = model_original.predict(X_test)
default_accuracy_original = accuracy_score(y_test, y_pred_original)
print(f"Default Accuracy on Original Data: {default_accuracy_original}")

# Confidence interval for default accuracy on original data
n_correct_default_original = default_accuracy_original * n_samples_test
ci_default_original = proportion_confint(count=n_correct_default_original, nobs=n_samples_test, alpha=0.05, method='normal')
print(f"95% confidence interval for default accuracy on original data: {ci_default_original}")

# Cross-validation on resampled data
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_resampled = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='accuracy')

# Calculate mean and standard deviation of CV accuracy per fold for resampled data
mean_cv_accuracy_per_fold_resampled = cv_scores_resampled.mean(axis=0)
std_cv_accuracy_per_fold_resampled = cv_scores_resampled.std(axis=0)

# Print CV results for resampled data
print("Mean CV Accuracy per Fold on Resampled Data:", cv_scores_resampled)
print("Standard Deviation of CV Accuracy per Fold on Resampled Data:", np.zeros(len(cv_scores_resampled)))  # The standard deviation per fold is 0 for individual folds
print("Overall Mean CV Accuracy on Resampled Data:", mean_cv_accuracy_per_fold_resampled)
print("Overall Std Deviation of CV Accuracy on Resampled Data:", std_cv_accuracy_per_fold_resampled)

# Confidence interval for mean CV accuracy on resampled data
n_correct_cv_resampled = mean_cv_accuracy_per_fold_resampled * n_samples_test
ci_cv_resampled = proportion_confint(count=n_correct_cv_resampled, nobs=n_samples_test, alpha=0.05, method='normal')
print(f"95% confidence interval for mean CV accuracy on resampled data: {ci_cv_resampled}")

# Cross-validation on original data
cv_scores_original = cross_val_score(model_original, X_train, y_train, cv=cv, scoring='accuracy')

# Calculate mean and standard deviation of CV accuracy per fold for original data
mean_cv_accuracy_per_fold_original = cv_scores_original.mean(axis=0)
std_cv_accuracy_per_fold_original = cv_scores_original.std(axis=0)

# Print CV results for original data
print("Mean CV Accuracy per Fold on Original Data:", cv_scores_original)
print("Standard Deviation of CV Accuracy per Fold on Original Data:", np.zeros(len(cv_scores_original)))  # The standard deviation per fold is 0 for individual folds
print("Overall Mean CV Accuracy on Original Data:", mean_cv_accuracy_per_fold_original)
print("Overall Std Deviation of CV Accuracy on Original Data:", std_cv_accuracy_per_fold_original)

# Confidence interval for mean CV accuracy on original data
n_correct_cv_original = mean_cv_accuracy_per_fold_original * n_samples_test
ci_cv_original = proportion_confint(count=n_correct_cv_original, nobs=n_samples_test, alpha=0.05, method='normal')
print(f"95% confidence interval for mean CV accuracy on original data: {ci_cv_original}")

# Bootstrap cross-validation on resampled data
n_iterations = 1000
n_size = int(len(X_train_resampled) * 0.8)
accuracy_scores_resampled = []

for i in range(n_iterations):
    indices = np.random.choice(len(X_train_resampled), n_size, replace=True)
    X_bootstrap, y_bootstrap = X_train_resampled[indices], y_train_resampled[indices]
    model.fit(X_bootstrap, y_bootstrap)
    y_pred_bootstrap = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_bootstrap)
    accuracy_scores_resampled.append(accuracy)

# Calculate mean and standard deviation of accuracy scores for resampled data
mean_accuracy_resampled = np.mean(accuracy_scores_resampled)
std_accuracy_resampled = np.std(accuracy_scores_resampled)

print(f"Bootstrap Accuracy on Resampled Data: Mean = {mean_accuracy_resampled:.4f}, Std = {std_accuracy_resampled:.4f}")

# Permutation test on resampled data
score_resampled, permutation_scores_resampled, p_value_resampled = permutation_test_score(
    model, X_train_resampled, y_train_resampled, cv=StratifiedKFold(5), n_permutations=1000, scoring='accuracy', n_jobs=-1
)

print(f"Permutation test score on resampled data: {score_resampled:.4f}, p-value: {p_value_resampled:.4f}")

# Calculate mean and standard deviation of permutation scores for resampled data
mean_permutation_scores_resampled = np.mean(permutation_scores_resampled)
std_permutation_scores_resampled = np.std(permutation_scores_resampled)

print("Mean Permutation Accuracy on Resampled Data:", mean_permutation_scores_resampled)
print("Standard Deviation of Permutation Accuracy on Resampled Data:", std_permutation_scores_resampled)
print("Overall Mean Permutation Accuracy on Resampled Data:", mean_permutation_scores_resampled)
print("Overall Std Deviation of Permutation Accuracy on Resampled Data:", std_permutation_scores_resampled)

# Plot bootstrap results for resampled data
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(accuracy_scores_resampled, kde=False)
plt.title('Bootstrap Accuracy Scores')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')

# Plot permutation results for resampled data
plt.subplot(1, 2, 2)
sns.histplot(permutation_scores_resampled, kde=False)
plt.axvline(score_resampled, color='r', linestyle='--')
plt.title('Permutation Accuracy Scores')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Generate a classification report for GaussianNB on resampled data
print("\nGaussianNB Classification Report on Resampled Data:")
class_report_resampled = classification_report(y_test, y_pred_resampled, target_names=label_encoder.classes_, output_dict=True)
class_report_df_resampled = pd.DataFrame(class_report_resampled).transpose()
print(class_report_df_resampled)

# Generate a classification report for GaussianNB on original data
print("\nGaussianNB Classification Report on Original Data:")
class_report_original = classification_report(y_test, y_pred_original, target_names=label_encoder.classes_, output_dict=True)
class_report_df_original = pd.DataFrame(class_report_original).transpose()
print(class_report_df_original)
