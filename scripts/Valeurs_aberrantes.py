import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

def manage_outliers(df, cols, method='all', contamination=0.05):
    '''
    This function detects and manages outliers in a our dataset using various methods.
    
    Parameters:
    -----------
    df: pandas.DataFrame
        The input dataset, here our openfoodfacts dataset with outliers
        
    cols: list
        The list of columns to apply outlier detection and management on
        
    method: str, optional (default='all')
        The outlier detection and management method to use. Available options are 'zscore', 'iqr', 'dbscan', 
        'isolationforest', and 'all' (default). 
        
    contamination: float, optional (default=0.05)
        The proportion of outliers expected in the dataset. Only applicable for DBSCAN and Isolation Forest methods.
    
    Returns:
    --------
    pandas.DataFrame
        The cleaned dataset with outliers managed
    '''
    
    # 1. Z-score method
    def zscore_method(df, cols):
        for col in cols:
            z = np.abs((df[col] - df[col].mean()) / df[col].std())
            df = df[z < 3]
        return df
    
    # 2. Interquartile range (IQR) method
    def iqr_method(df, cols):
        for col in cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        return df
    
    # 3. DBSCAN method
    def dbscan_method(df, cols, contamination):
        scaler = StandardScaler()
        X = scaler.fit_transform(df[cols])
        dbscan = DBSCAN(eps=0.5, min_samples=5, metric='euclidean', n_jobs=-1)
        dbscan.fit(X)
        mask = dbscan.labels_ != -1
        df = df[mask]
        return df
    
    # 4. Isolation Forest method
    def isolationforest_method(df, cols, contamination):
        scaler = StandardScaler()
        X = scaler.fit_transform(df[cols])
        isolationforest = IsolationForest(n_estimators=100, max_samples='auto', contamination=contamination, n_jobs=-1)
        isolationforest.fit(X)
        mask = isolationforest.predict(X) != -1
        df = df[mask]
        return df
    
    # Apply the specified method or all methods
    if method == 'zscore':
        df = zscore_method(df, cols)
    elif method == 'iqr':
        df = iqr_method(df, cols)
    elif method == 'dbscan':
        df = dbscan_method(df, cols, contamination)
    elif method == 'isolationforest':
        df = isolationforest_method(df, cols, contamination)
    else:
        df = zscore_method(df, cols)
        df = iqr_method(df, cols)
        df = dbscan_method(df, cols, contamination)
        df = isolationforest_method(df, cols, contamination)
    
    return df
