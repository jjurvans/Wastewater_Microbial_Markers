#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 14:02:24 2024

@author: jaanajurvansuu
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, permutation_test_score, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
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

# Train GaussianNB model and calculate default accuracy
model = GaussianNB()
model.fit(X_train_resampled, y_train_resampled)
y_pred = model.predict(X_test)
default_accuracy = accuracy_score(y_test, y_pred)
print(f"Default Accuracy: {default_accuracy}")

# Confidence interval for default accuracy
n_correct_default = default_accuracy * n_samples_test
ci_default = proportion_confint(count=n_correct_default, nobs=n_samples_test, alpha=0.05, method='normal')
print(f"95% confidence interval for default accuracy: {ci_default}")

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='accuracy')

# Calculate mean and standard deviation of CV accuracy per fold
mean_cv_accuracy_per_fold = cv_scores.mean(axis=0)
std_cv_accuracy_per_fold = cv_scores.std(axis=0)

# Print CV results
print("Mean CV Accuracy per Fold:", cv_scores)
print("Standard Deviation of CV Accuracy per Fold:", np.zeros(len(cv_scores)))  # The standard deviation per fold is 0 for individual folds
print("Overall Mean CV Accuracy:", mean_cv_accuracy_per_fold)
print("Overall Std Deviation of CV Accuracy:", std_cv_accuracy_per_fold)

# Confidence interval for mean CV accuracy
n_correct_cv = mean_cv_accuracy_per_fold * n_samples_test
ci_cv = proportion_confint(count=n_correct_cv, nobs=n_samples_test, alpha=0.05, method='normal')
print(f"95% confidence interval for mean CV accuracy: {ci_cv}")

# Bootstrap cross-validation
n_iterations = 1000
n_size = int(len(X_train_resampled) * 0.8)
accuracy_scores = []

for i in range(n_iterations):
    indices = np.random.choice(len(X_train_resampled), n_size, replace=True)
    X_bootstrap, y_bootstrap = X_train_resampled[indices], y_train_resampled[indices]
    model.fit(X_bootstrap, y_bootstrap)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

# Calculate mean and standard deviation of accuracy scores
mean_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)

print(f"Bootstrap Accuracy: Mean = {mean_accuracy:.4f}, Std = {std_accuracy:.4f}")

# Permutation test
score, permutation_scores, p_value = permutation_test_score(
    model, X_train_resampled, y_train_resampled, cv=StratifiedKFold(5), n_permutations=1000, scoring='accuracy', n_jobs=-1
)

print(f"Permutation test score: {score:.4f}, p-value: {p_value:.4f}")

# Calculate mean and standard deviation of permutation scores
mean_permutation_scores = np.mean(permutation_scores)
std_permutation_scores = np.std(permutation_scores)

print("Mean Permutation Accuracy:", mean_permutation_scores)
print("Standard Deviation of Permutation Accuracy:", std_permutation_scores)
print("Overall Mean Permutation Accuracy:", mean_permutation_scores)
print("Overall Std Deviation of Permutation Accuracy:", std_permutation_scores)

# Plot bootstrap results
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(accuracy_scores, kde=True)
plt.title('Bootstrap Accuracy Scores')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')

# Plot permutation results
plt.subplot(1, 2, 2)
sns.histplot(permutation_scores, kde=True)
plt.axvline(score, color='r', linestyle='--')
plt.title('Permutation Accuracy Scores')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Generate a classification report for GaussianNB
print("\nGaussianNB Classification Report:")
class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
class_report_df = pd.DataFrame(class_report).transpose()
print(class_report_df)

# Train and evaluate a baseline model (DummyClassifier)
baseline_model = DummyClassifier(strategy="most_frequent")
baseline_model.fit(X_train_resampled, y_train_resampled)
baseline_y_pred = baseline_model.predict(X_test)
baseline_accuracy = accuracy_score(y_test, baseline_y_pred)
print(f"\nBaseline Accuracy: {baseline_accuracy}")

# Generate a classification report for the baseline model
print("\nBaseline Model Classification Report:")
baseline_class_report = classification_report(y_test, baseline_y_pred, target_names=label_encoder.classes_, output_dict=True)
baseline_class_report_df = pd.DataFrame(baseline_class_report).transpose()
print(baseline_class_report_df)
