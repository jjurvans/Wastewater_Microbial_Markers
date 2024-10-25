#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 14:04:57 2024

@author: jaanajurvansuu
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize

# Load data
df = pd.read_csv('CodeEffsDataframe.csv')

# Separate features and target
X = df.drop('SAMPLE', axis=1)
y = df['SAMPLE']

# Encode target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_binary = label_binarize(y_encoded, classes=[0, 1, 2, 3, 4, 5])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Ensure the target split is correctly stratified
print("Train target distribution:", np.bincount(y_train))
print("Test target distribution:", np.bincount(y_test))

# Models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
    'Support Vector Machine': SVC(random_state=42, probability=True, class_weight='balanced'),
    'k-Nearest Neighbors': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced')
}

# SMOTE sampling strategy range for multi-class
smote_ratios = np.arange(0.0, 0.5, 0.1)  # Using numpy to generate float range
k_values = range(10, 11)

# Store metrics for different SMOTE ratios and k-values
metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
results = {metric: {model_name: np.zeros((len(k_values), len(smote_ratios))) for model_name in models.keys()} for metric in metrics}

# Function to create sampling strategy dictionary and set k_neighbors dynamically
def create_smote_instance(y_train, ratio):
    class_counts = np.bincount(y_train)
    majority_class = np.max(class_counts)
    sampling_strategy = {class_idx: max(class_counts[class_idx], int(majority_class * ratio)) for class_idx in range(len(class_counts))}
    
    # Determine the number of samples in the smallest class in y_train
    minority_class_count = np.min(np.bincount(y_train))
    k_neighbors = max(1, min(5, minority_class_count - 1))
    
    return SMOTE(random_state=42, sampling_strategy=sampling_strategy, k_neighbors=k_neighbors)

# Loop over k-values, models, and SMOTE ratios
for i, k in enumerate(k_values):
    print(f"Evaluating models with k={k} features...")
    
    # Select top k features
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    for model_name, model in models.items():
        print(f"Optimizing {model_name}...")

        # Create a pipeline with SMOTE and the classifier
        pipeline = Pipeline([
            ('smote', SMOTE(random_state=42)),  # Placeholder for pipeline
            ('classifier', model)
        ])

        for j, ratio in enumerate(smote_ratios):
            smote_instance = create_smote_instance(y_train, ratio)
            pipeline.set_params(smote=smote_instance)
            pipeline.fit(X_train_selected, y_train)
            y_pred = pipeline.predict(X_test_selected)
            
        
            # Ensure y_prob is computed only if the model supports predict_proba
            y_prob = pipeline.predict_proba(X_test_selected) if hasattr(pipeline, "predict_proba") else None

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            if y_prob is not None and len(np.unique(y_test)) > 1:
                try:
                    auc = roc_auc_score(y_binary[:len(y_prob)], y_prob, average='weighted', multi_class='ovr')
                except ValueError:
                    auc = np.nan
            else:
                auc = np.nan

            results['accuracy'][model_name][i, j] = accuracy
            results['precision'][model_name][i, j] = precision
            results['recall'][model_name][i, j] = recall
            results['f1_score'][model_name][i, j] = f1
           

            print(f"Metrics for {model_name} with k={k} and SMOTE ratio {ratio}: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, AUC: {auc:.4f}")

# Identify and print the best metric values for each model
best_metrics = {metric: {} for metric in metrics}
for model_name in models.keys():
    for metric in metrics:
        best_value = -np.inf
        best_k = None
        best_ratio = None
        for i, k in enumerate(k_values):
            for j, ratio in enumerate(smote_ratios):
                if results[metric][model_name][i, j] > best_value:
                    best_value = results[metric][model_name][i, j]
                    best_k = k
                    best_ratio = ratio
                elif results[metric][model_name][i, j] == best_value:
                    if k < best_k or (k == best_k and ratio < best_ratio):
                        best_k = k
                        best_ratio = ratio
        best_metrics[metric][model_name] = (best_value, best_k, best_ratio)

# Print the best metrics for each model
for metric in metrics:
    print(f"\nBest {metric.capitalize()} Values:")
    for model_name, (value, best_k, best_ratio) in best_metrics[metric].items():
        print(f"{model_name}: {value:.4f} (k={best_k}, SMOTE ratio={best_ratio})")
