import tach1_0_anas as tach1_0

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction import FeatureHasher
# from category_encoders import CountEncoder, QuantileEncoder
from scipy.sparse import save_npz, load_npz

# Call the get_data() function
# data = dl.get_data(file_path='../data/en.openfoodfacts.org.products.csv', nrows=50)
# print(tach1_0.data.head())
data_cleaned = tach1_0.data_with_less_30_nan(tach1_0.data) 
print(data_cleaned.info())
# Supprimer les catégories rares
def supprimer_categories_rares(df, seuil):
    for col in df.columns:
        if df[col].dtype == 'object':
            freq = df[col].value_counts(normalize=True)
            rare_categories = freq[freq < seuil].index
            df[col] = df[col].apply(lambda x: 'Autre' if x in rare_categories else x)
    return df

data = supprimer_categories_rares(data_cleaned, 0.01)
print(data.head())

# Encoder les features catégorielles avec OneHotEncoder et Sparse Matrix
def one_hot_encoder(df):
    ohe = OneHotEncoder(sparse=True)
    for col in df.columns:
        if df[col].dtype == 'object':
            encoded_features = ohe.fit_transform(df[[col]])
            save_npz(f'{col}_ohe.npz', encoded_features)
            df = df.drop(col, axis=1)
    return df

data_ohe = one_hot_encoder(data_cleaned)
print(data_ohe.head())
exit()
# Utiliser le Feature Hashing
def feature_hashing(df, n_features):
    fh = FeatureHasher(n_features=n_features, input_type='string')
    for col in df.columns:
        if df[col].dtype == 'object':
            hashed_features = fh.fit_transform(df[col])
            df = df.drop(col, axis=1)
            for i in range(n_features):
                df[f'{col}_hashed_{i}'] = hashed_features[:, i].toarray().ravel()
    return df

data_fh = feature_hashing(data, 10)

# Utiliser CountEncoder
def count_encoder(df):
    ce = CountEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = ce.fit_transform(df[col])
    return df

data_ce = count_encoder(data)

# Utiliser QuantileEncoder
def quantile_encoder(df):
    qe = QuantileEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = qe.fit_transform(df[col])
    return df

data_qe = quantile_encoder(data)