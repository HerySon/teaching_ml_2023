import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer

def drop_columns_with_missing_values(data, threshold=0.3):
    """
    Drops columns that have more than the specified threshold of missing values.

    Parameters:
    data (DataFrame): the dataset to clean
    threshold (float): the threshold above which a column will be dropped. Default is 0.3

    Returns:
    DataFrame: the cleaned dataset
    """
    # Compute the percentage of missing values for each column
    missing_percentage = data.isnull().sum() / data.shape[0]

    # Select columns to drop
    cols_to_drop = missing_percentage[missing_percentage > threshold].index

    # Drop the selected columns
    cleaned_data = data.drop(columns=cols_to_drop)

    return cleaned_data


def impute_missing_values(data, columns=None, missing_values='NaN', n_neighbors=5, weights='uniform'):
    """
    Imputes missing values using the K-Nearest Neighbor algorithm with specified settings.

    Parameters:
    data (DataFrame): the dataset to clean
    columns (list): list of columns to impute missing values in. Default is None, in which case all columns are imputed.
    missing_values (float or str): the placeholder for missing values. Default is 'NaN'.
    n_neighbors (int): the number of neighbors to use. Default is 5.
    weights (str or callable): the weight function used in prediction. Default is 'uniform'.

    Returns:
    DataFrame: the cleaned dataset
    """
    # If columns are not specified, impute all columns
    if columns is None:
        columns = data.columns

    # Check if columns are numeric
    if not all(data[col].dtype.kind in ['i', 'u', 'f'] for col in columns):
        raise ValueError('All columns to be imputed must be numeric.')

    # Create an instance of the KNNImputer class with specified parameters
    imputer = KNNImputer(missing_values=missing_values, n_neighbors=n_neighbors, weights=weights)

    # Impute missing values
    imputed_data = data.copy()
    imputed_data[columns] = imputer.fit_transform(imputed_data[columns])

    return imputed_data


def univariate_imputation(data, columns=None, strategy='mean', missing_values=np.nan):
    """
    Imputes missing values using the specified univariate imputation strategy.

    Parameters:
    data (DataFrame): the dataset to clean
    columns (list): list of columns to impute missing values in. Default is None, in which case all columns are imputed.
    strategy (str): the imputation strategy. Default is 'mean'.
    missing_values (float or str): the placeholder for missing values. Default is np.nan.

    Returns:
    DataFrame: the cleaned dataset
    """
    # If columns are not specified, impute all columns
    if columns is None:
        columns = data.columns

    # Check if columns are numeric
    if not all(data[col].dtype.kind in ['i', 'u', 'f'] for col in columns):
        raise ValueError('All columns to be imputed must be numeric.')

    # Create an instance of the SimpleImputer class with specified parameters
    imputer = SimpleImputer(missing_values=missing_values, strategy=strategy)

    # Impute missing values
    imputed_data = data.copy()
    imputed_data[columns] = imputer.fit_transform(imputed_data[columns])

    return imputed_data
