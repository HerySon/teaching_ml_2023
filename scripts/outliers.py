import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

def outliers_process(df, columns, method = 'nan', k=1.5, sklearn_method=False):
    """
    Detects and handles outliers in a pandas DataFrame using Interquartile Range or
    IsolationForest detection method.
    If IQR method, mean or median calculated on all values in column.
    If IsolationForest method, mean or median calculated on all non-outlier values in column.

    Keyword arguments:
    ------------------
        df: pandas.DataFrame
            The input DataFrame.
        columns: list of str
            The list of column names to handle outliers for.
        method: str, optional (default 'nan').
            The method to use for handling outliers.
            Available methods are 'nan', mean', 'median', and 'drop'
        k: float, optional (default 1.5)
            The multiplier for the IQR range.
        sklearn_method: bool, optional (default False)
            Will use IQR method to detect and remove outliers if set False,
            else will use IsolationForest method.

    Returns:
    --------
        pandas.DataFrame
            The DataFrame with outliers handled.

    Author:
    -------
        Joëlle Sabourdy
    """
    df_outliers = df.copy()

    if sklearn_method == True:
        for i in columns:
            # Extract values to process
            X = df_outliers[[i]].values
        
            # Define and fit the IsolationForest model
            model = IsolationForest()
            model.fit(X)
        
            # Compute the outlier scores and classify outliers
            outlier_scores = model.decision_function(X)
            outliers_condition = model.predict(X) == -1
            
            # handle outliers
                # replace outliers with NaN value
            if method == 'nan':
                df_outliers.loc[outliers_condition, i] = np.nan
                # drop rows with outliers
            elif method == 'drop':
                df_outliers = df_outliers.loc[~outliers_condition, :]
                 # replace outliers with median value
            elif method == 'median':
                median = df_outliers.loc[~outliers_condition, i].median()
                df_outliers.loc[outliers_condition, i] = median
                 # replace outliers with mean value
            elif method == 'mean':
                mean = df_outliers.loc[~outliers_condition, i].mean()
                df_outliers.loc[outliers_condition, i] = mean
            else:
                raise ValueError("Invalid method. Allowed methods: 'nan','drop', 'median', 'mean'.")
        
    else:
        # InterQuartile Range method
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
    
        # Calculate lower and upper bounds for outliers detection
        min = Q1 - k * IQR
        max = Q3 + k * IQR
    
        # conditions
        outliers_condition = ((df_outliers < min) | (df_outliers > max))
        
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
            raise ValueError("Invalid method. Allowed methods: 'nan','drop', 'median', 'mean'.")
            
    return df_outliers



if __name__ == "__main__":
    # Consider dataset containing ramen rating
    df = pd.DataFrame({
        'brand': ['Yum Yum', 'Yum Yum', 'Indomie', 'Indomie', 'Tanoshi', 'Cup Noodles'],
        'style': ['cup', 'cup', 'cup', 'pack', 'pack', 'cup'],
        'rating': [49, 4, 3.5, 1, 5, 2],
        'grams': [80, 80, 80, 90, 90, 500]
        })
    # Define numeric features to handle outliers
    columns = ['rating', 'grams']
    # Handle outliers in numeric features
    df_out = outliers_process(df, columns, method = 'mean', k=1.5, sklearn_method=True)
    print(df_out)