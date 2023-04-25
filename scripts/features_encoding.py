import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


def non_numeric_features_encoder(df, columns, encoder_type=OrdinalEncoder, sparse=False):
    """
    Encode non-numeric features in a pandas DataFrame using OrdinalEncoder or
    OneHotEncoder from Scikit-Learn.

    Keyword arguments:
    ------------------
        df: pandas.DataFrame
            The input DataFrame.
        columns: list of str
            The list of column names to encode.
        encoder_type: type, optional (default OrdinalEncoder)
            The method to encode features. Possible values are: 
            - OrdinalEncoder: The features are converted to ordinal integers. This
            results in a single column of integers (0 to n_categories - 1) per feature.
            - OneHotEncoder: This creates a binary column for each category and returns
            a sparse matrix or dense array (depending on the sparse_output parameter).
        sparse: bool, optional (default False)
            Will return sparse matrix if set True else will return an array.

    Returns:
    --------
        pandas.DataFrame
            The encoded pandas DataFrame if sparse=False.
        sparse matrix and encoder 
            If encoder_type=OneHotEncoder and sparse=True.

    Author:
    -------
        Joëlle Sabourdy
    """
    # create a OrdinalEncoder/OneHotEncoder object
    if encoder_type == OrdinalEncoder:
        encoder = encoder_type()
    elif encoder_type == OneHotEncoder:
        encoder = encoder_type(sparse_output=sparse)
            
    # encode features selected
    encoded_features = encoder.fit_transform(df[columns])
    
    if encoder_type == OneHotEncoder and sparse == True:
        # return sparse matrix and encoder object
        return encoded_features, encoder
    else:
        # create a list of the new features names
        df[encoder.get_feature_names_out()] = encoded_features

        # drop original features if OneHotEncoder
        if encoder_type == OneHotEncoder:
            df.drop(columns, axis=1, inplace=True)
    
        # return new DataFrame with features encoded
        return df



def concat_matrix(df, columns, matrix, encoder):
    """
    Concat DataFrame and sparse matrix from non_numeric_features_encoder function,
    if encoder_type=OneHotEncoder and sparse=True.

    Keyword arguments:
    ------------------
        df: pandas.DataFrame
            The input DataFrame.
        columns: list of str
            The list of column names to encode.
        matrix: sparse matrix
            The sparse matrix output of non_numeric_features_encoder() function.
        encoder: encoder object
            The encoder object from non_numeric_features_encoder() function.

    Returns:
    --------
        pandas.DataFrame
            The concatenated pandas DataFrame with features encoded.

    Author:
    -------
        Joëlle Sabourdy
    """
    import scipy.sparse
    # matrix to pd
    df_matrix = pd.DataFrame.sparse.from_spmatrix(matrix)
    # create a list of the new features names
    df[encoder.get_feature_names_out()] = df_matrix
    # drop original features if OneHotEncoder
    df.drop(columns, axis=1, inplace=True)
    # return new DataFrame with features encoded
    return df



if __name__ == "__main__":
    # Consider dataset containing ramen rating
    df = pd.DataFrame({
        'brand': ['Yum Yum', 'Yum Yum', 'Indomie', 'Indomie', 'Tanoshi', 'Cup Noodles'],
        'style': ['cup', 'cup', 'cup', 'pack', 'pack', 'cup'],
        'rating': [49, 4, 3.5, 1, 5, 2],
        'grams': [80, 80, 80, 90, 90, 500]
        })
    # Define non-numeric features to encode
    columns = ['brand', 'style']
    # Encode non-numeric features and get sparse matrix
    matrix, encoder = non_numeric_features_encoder(df, columns, encoder_type=OneHotEncoder, sparse=True)
    # Concat DataFrame and sparse matrix
    df_mat = concat_matrix(df, columns, matrix, encoder)
    print(df_mat)