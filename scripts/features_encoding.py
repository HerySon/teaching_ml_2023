import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


def non_numeric_features_encoder(df, columns, encoder_type=OrdinalEncoder, sparse=False):
    """
    Encode non-numeric features in a pandas dataframe using OrdinalEncoder or OneHotEncoder from Scikit-Learn.   
    Args:
        df : pandas dataframe (The input dataframe)
        columns : list of str (The list of column names to encode)
        encoder_type : OrdinalEncoder or OneHotEncoder (Default='OrdinalEncoder')
            'OrdinalEncoder' : The features are converted to ordinal integers. This results in a single column of integers (0 to n_categories - 1) per feature.
            'OneHotEncoder' : This creates a binary column for each category and returns a sparse matrix or dense array (depending on the sparse_output parameter).
        sparse : bool (Default=False), use for OneHotEncoder method
            Will return sparse matrix if set True else will return an array.
    Returns:
        df : the encoded dataframe
    """
    # create a OrdinalEncoder/OneHotEncoder object
    if encoder_type == OrdinalEncoder:
        encoder = encoder_type()
    elif encoder_type == OneHotEncoder:
        encoder = encoder_type(sparse_output=sparse)
        
    # encode features selected
    encoded_features = encoder.fit_transform(df[columns])
    
    # create a list of the new features names or return matrix
    if encoder_type == OneHotEncoder and sparse == True:
        return encoded_features
    else:
        df[encoder.get_feature_names_out()] = encoded_features

    # drop original features if OneHotEncoder
    if encoder == OneHotEncoder:
        df.drop(columns, axis=1, inplace=True)
    
    # return new dataframe with features encoded
    return df




# Test

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