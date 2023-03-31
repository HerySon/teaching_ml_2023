import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def non_numeric_features_encoder(df, columns):
    """
    Encode non-numeric features in a pandas dataframe using OneHotEncoder from Scikit-Learn.
    Args:
        df : pandas dataframe (The input dataframe)
        columns : list of str (The list of column names to encode)
    Returns:
        df : the encoded dataframe
    """
    # create a OneHotEncoder object
    encoder = OneHotEncoder(sparse=False)
    # encode features selected
    encoded_features = encoder.fit_transform(df[columns])
    # create a list of the new features names
    df[encoder.get_feature_names_out()] = encoded_features
    # drop original features
    df.drop(columns, axis=1, inplace=True)
    # return new dataframe with features encoded
    return df
