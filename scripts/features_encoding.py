import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder 


def non_numeric_features_encoder(df, columns):
    """
    Encode non-numeric features in a pandas dataframe using OrdinalEncoder from Scikit-Learn.
    The input to this transformer should be an array-like of integers or strings, denoting the values taken on by categorical (discrete) features.
    The features are converted to ordinal integers. This results in a single column of integers (0 to n_categories - 1) per feature.
    Args:
        df : pandas dataframe (The input dataframe)
        columns : list of str (The list of column names to encode)
    Returns:
        df : the encoded dataframe
    """
    # create a OrdinalEncoder object
    encoder = OrdinalEncoder()
    # encode features selected
    encoded_features = encoder.fit_transform(df[columns])
    # create a list of the new features names
    df[encoder.get_feature_names_out()] = encoded_features
    # return new dataframe with features encoded
    return df




# Unit test

# Consider dataset containing ramen rating
df = pd.DataFrame({
    'brand': ['Yum Yum', 'Yum Yum', 'Indomie', 'Indomie', 'Tanoshi', 'Cup Noodles'],
    'style': ['cup', 'cup', 'cup', 'pack', 'pack', 'cup'],
    'rating': [49, 4, 3.5, 1, 5, 2],
    'grams': [80, 80, 80, 90, 90, 500]
    })

# Define non-numeric features to encode
columns = ['brand', 'style']

# Encore non-numeric features. Converted to ordinal integers (0 to n_categories - 1)
non_numeric_features_encoder(df, columns)