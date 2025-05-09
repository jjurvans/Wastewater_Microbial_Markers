# Wastewater_Microbial_Markers
Machine Learning-Based Identification of Wastewater Treatment Plant-Specific Microbial Markers Using 16S rRNA Sequencing

This project includes the code used in the machine learning part of the article. The input data used in each code set is based on Qiime2 feature-table on which the relative abundance of all bacteria and archaea calculated for each wastewater treatment plant samples (CodeEffsDataframe.csv). The raw sequences are available from the NCBI SRA under BioProject ID PRJNA1177763, with accessions SAMN44450032–SAMN44450088.

The repository contains:
1. CodeEffsDataframe.csv - data frame for the relative abundance of all bacteria genus found from the six studied wastewater treatment plants.
2. KvalueSMOTEandModelsOptimising.py - Optimising model (Random Forest, Gradient Boosting, Logistic Regression, Support Vector Machine, k-Nearest Neighbors, Naive Bayes and Decision Tree), k-value and SMOTE ratio for the data.
3. SHAP.py - SHapley Additive exPlanations summary plots for each wastewater treatment plants.
4. GaussianNB_Report_CrossValidation.py - Classification report for Naive Bayes model with ten selected features and SMOTE ratio of 0.4. Bootstrap and permutation test accuracy distributions for Gaussian Naive Bayes Classifier with optimised feature number of ten and SMOTE ratio of 0.4.
5. GaussianNBProbabilityContours.py - PCA Plot with Gaussian Naive Bayes decision boundaries and uncertainty contours.
