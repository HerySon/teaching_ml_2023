import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score  
from sklearn.model_selection import ParameterGrid

def train_and_optimize_DBSCAN(data, feature_cols, scale_data=False,
                              param_grid={'eps': [0.1, 0.5, 1.0], 
                                          'min_samples': [2, 5, 10]}):
    """
    --------------------------------------------------------------------------
    Trains and optimizes DBSCAN on a given dataset.
    -----------------------------------------------------------------------------
    
    Parameters:
    ----------------------------------------------------------------------------
        - data (pandas.DataFrame): 
            The dataset to be used for training and optimization.
        - feature_cols (list): 
            A list of column names for the features to be used for training.
        - scale_data (bool):
            Whether to scale the data or not. Default is False.
        - param_grid (dict):
            The grid of parameters to search over. Default is {'eps': [0.1, 0.5, 1.0], 'min_samples': [2, 5, 10]}.
    -----------------------------------------------------------------------------------
    Returns:
    -----------------------------------------------------------------------------------
        A tuple of the optimal values for eps and min_samples, 
        as determined by the silhouette score.
    -------------------------------------------------------------------------------
    """

    # Select the feature columns
    X = data[feature_cols].values
    
    # Normalize the data if requested
    if scale_data:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # Try different values of eps and min_samples and evaluate them
    scores = []
    for params in ParameterGrid(param_grid):
        dbscan = DBSCAN(**params)
        dbscan.fit(X)
        if len(set(dbscan.labels_)) > 1:  # Skip if only one cluster is found
            score = silhouette_score(X, dbscan.labels_)
            scores.append((params['eps'], params['min_samples'], score))
            print("DBSCAN with eps={}, min_samples={} - Silhouette Score: {}".format(params['eps'], params['min_samples'], score))
    
    # Find the optimal values for eps and min_samples
    if len(scores) > 0:
        optimal_eps, optimal_min_samples, _ = max(scores, key=lambda x: x[2])
        print("Optimal values - eps: {}, min_samples: {}".format(optimal_eps, optimal_min_samples))
        return (optimal_eps, optimal_min_samples)
    else:
        print("Could not find optimal values.")
        return None

  
if __name__ == "__main__":  
    train_and_optimize_DBSCAN(data, feature_cols, scale_data=True, param_grid={'eps': [0.1, 0.5, 1.0], 'min_samples': [2, 5, 10], 'metric': ['euclidean', 'manhattan']})
