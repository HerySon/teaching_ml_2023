import numpy as np
import pandas as pd

def outliers_process(df, columns, method = 'nan', k=1.5, sklearn_method=None):
    """
    Detects and handles outliers in a pandas dataframe using Interquartile Range.
    Args:
        df : pandas dataframe (The input dataframe)
        columns : list of str (The list of column names to handle outliers for)
        method : str (default='nan').
            The method to use for handling outliers. Available methods are 'nan', mean', 'median', and 'drop'
        k : float (Default=1.5)
            The multiplier for the IQR range
        sklearn_method: str, optional (default=None)
            'IsolationForest' method from Scikit-learn is use for detecting outliers
    Returns:
        df : the dataframe with outliers handled
    """
    df_outliers = df.copy()

    if sklearn_method == 'IsolationForest':
        # Fit the IsolationForest estimator
        isof = IsolationForest()
        isof.fit(df_outliers[columns])
        outlier_pred = isof.predict(df_outliers[columns])
        outliers_condition = outlier_pred == -1
        
    else:
        # InterQuartile Range method
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
    
        # Calculate lower and upper bounds for outliers detection
        min = Q1 - k * IQR
        max = Q3 + k * IQR
    
        # conditions
        outliers_condition = ((df_outliers[columns] < min) | (df_outliers[columns] > max))

    # handle outliers
        # replace outliers with NaN value
    if method == 'nan':
        df_outliers = df_outliers.mask(outliers_condition)
        # drop rows with outliers
    elif method == 'drop':
        df_outliers = df_outliers[~outliers_condition.any(axis=1)]
        # replace outliers with median value
    elif method == 'median':
        median = df_outliers[columns].median()
        df_outliers = df_outliers.mask(outliers_condition, median, axis=1)
        # replace outliers with mean value
    elif method == 'mean':
        mean = df_outliers[columns].mean()
        df_outliers = df_outliers.mask(outliers_condition, mean, axis=1)
    else:
        return df
    return df_outliers