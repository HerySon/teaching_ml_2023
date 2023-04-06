import pandas as pd
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