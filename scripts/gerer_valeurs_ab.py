import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

def manage_outliers(data, method='StandardScaler', threshold=3, strategy='median'):
    """
    ---------------------------------------------------------------
    Goal:
    Detects and manages outliers in a dataset using various methods.
    ---------------------------------------------------------------
    Parameters:
    -----------
    data : pandas.DataFrame or numpy.ndarray
        The dataset to manage outliers for.
    method : str, optional (default='StandardScaler')
        The method to use for outlier detection. Can be one of:
            - 'StandardScaler': Uses the StandardScaler method to scale the data and 
               then uses the z-score method based on the standard deviation.
            - 'DBSCAN': Uses the DBSCAN clustering algorithm to identify outliers.
            - 'IsolationForest': Uses the Isolation Forest algorithm to identify outliers.
    threshold : float, optional (default=3)
        The threshold used to determine outliers. 
        Values above or below this threshold are considered outliers.
    strategy : str, optional (default='median')
        The strategy used to manage outliers. Can be one of:
            - 'median': Replaces outliers with the median value of the feature.
            - 'mean': Replaces outliers with the mean value of the feature.
    ----------------------------------------------------------------------
    Returns:
    --------
    pandas.DataFrame
        The cleaned dataset with outliers managed.
    """
    # Scaling the data if method is StandardScaler
    if method == 'StandardScaler':
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    # Detecting outliers using the selected method
    if method == 'StandardScaler':
        z_scores = np.abs(data)
        outliers = np.where(z_scores > threshold)
    elif method == 'DBSCAN':
        dbscan = DBSCAN(eps=threshold, min_samples=2)
        outliers = dbscan.fit_predict(data) == -1
    elif method == 'IsolationForest':
        iso_forest = IsolationForest(contamination=threshold)
        outliers = iso_forest.fit_predict(data) == -1

    # Managing outliers using the selected strategy
    if strategy == 'median':
        replacements = np.median(data, axis=0)
    elif strategy == 'mean':
        replacements = np.mean(data, axis=0)

    data[outliers] = replacements

    # Returning the cleaned data
    if method == 'StandardScaler':
        data = scaler.inverse_transform(data)
    return pd.DataFrame(data)


if __name__ == "__main__":
    
    # Creating a simple dataframe with some outliers
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5, 100],
        'B': [2, 4, 6, 8, 10, 200],
        'C': [5, 10, 15, 20, 25, 1000]
    })

    # Using the manage_outliers() function to clean the dataframe
    clean_df = manage_outliers(df, method='IsolationForest', threshold=0.05, strategy='mean')

    # Printing the original dataframe and the cleaned dataframe
    print("Original Dataframe:\n", df)
    print("\nCleaned Dataframe:\n", clean_df)
