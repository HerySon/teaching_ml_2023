########################
#Selection des features#
########################
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, chi2, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
data = df
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Remove low variance features
var_threshold = 0.1
var_selector = VarianceThreshold(threshold=var_threshold)
X = var_selector.fit_transform(X)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Select features based on correlation
corr_threshold = 0.5
corr_matrix = X.corr().abs()
corr_features = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if corr_matrix.iloc[i, j] > corr_threshold:
            corr_features.add(corr_matrix.columns[i])

# Select features based on mutual information
mutual_info_k = 10
mutual_info_selector = SelectKBest(score_func=mutual_info_classif, k=mutual_info_k)
mutual_info_selector.fit(X, y)
mutual_info_scores = pd.DataFrame(mutual_info_selector.scores_, index=X.columns, columns=['Score'])
mutual_info_scores = mutual_info_scores.sort_values(by='Score', ascending=False)
mutual_info_features = set(mutual_info_scores.index[:mutual_info_k])

# Select features based on ANOVA F-value
f_k = 10
f_selector = SelectKBest(score_func=f_classif, k=f_k)
f_selector.fit(X, y)
f_scores = pd.DataFrame(f_selector.scores_, index=X.columns, columns=['Score'])
f_scores = f_scores.sort_values(by='Score', ascending=False)
f_features = set(f_scores.index[:f_k])

# Select features based on chi-squared test
chi_k = 10
chi_selector = SelectKBest(score_func=chi2, k=chi_k)
chi_selector.fit(X, y)
chi_scores = pd.DataFrame(chi_selector.scores_, index=X.columns, columns=['Score'])
chi_scores = chi_scores.sort_values(by='Score', ascending=False)
chi_features = set(chi_scores.index[:chi_k])

# Select features based on random forest importance
rf_n_estimators = 100
rf_max_depth = 5
rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth), threshold='mean')
rf_selector.fit(X, y)
rf_features = set(X.columns[rf_selector.get_support()])

# Select features based on logistic regression coefficients
lr_c = 0.1
lr_selector = SelectFromModel(LogisticRegression(penalty='l1', C=lr_c, solver='liblinear'), threshold=None)
lr_selector.fit(X, y)
lr_features = set(X.columns[lr_selector.get_support()])

# Combine selected features
selected_features = corr_features.union(mutual_info_features, f_features, chi_features, rf_features, lr_features)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X[list(selected_features)], y, test_size=0.2, random_state=42)

# Train and evaluate model using selected features
rf_n_estimators = 100
rf_max_depth = 5
model = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=42
