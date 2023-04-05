import numpy as np
import pandas as pd

def outliers_process(df, columns, method = 'nan', k=1.5):
    """
    Detects and handles outliers in a pandas dataframe using Interquartile Range.
    Args:
        df : pandas dataframe (The input dataframe)
        columns : list of str (The list of column names to handle outliers for))
        method : str, optional (default='nan').
            The method to use for handling outliers. Available methods are 'nan', mean', 'median', and 'drop'.
        k : float (The multiplier for the IQR range. Default is 1.5)
    Returns:
        df : the dataframe with outliers handled
    """
    df_outliers = df.copy()
    
    # InterQuartile Range method
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    
    # Calculate lower and upper bounds for outliers detection
    min = Q1 - 1.5 * IQR
    max = Q3 + 1.5 * IQR
    
    # handle outliers
        # replace outliers with NaN value
    if method == 'nan':
        df_outliers = df_outliers.mask(((df_outliers[columns] < min)) | ((df_outliers[columns] > max)))
        # drop rows with outliers
    elif method == 'drop':
        df_outliers = df_outliers[~((df_outliers[columns] < min) | (df_outliers[columns] > max)).any(axis=1)]
        # replace outliers with median value
    elif method == 'median':
        median = df_outliers[columns].median()
        df_outliers = df_outliers[columns].mask(((df_outliers[columns] < min)) | ((df_outliers[columns] > max)), median, axis=1)
        # replace outliers with mean value
    elif method == 'mean':
        mean = df_outliers[columns].mean()
        df_outliers = df_outliers[columns].mask(((df_outliers[columns] < min)) | ((df_outliers[columns] > max)), mean, axis=1)
    else:
        return df
    return df_outliers