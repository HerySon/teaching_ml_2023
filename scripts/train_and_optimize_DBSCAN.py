import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def train_and_optimize_DBSCAN(data, feature_cols, eps_min=0.1, eps_max=1.0, eps_step=0.1, min_samples_min=2, min_samples_max=10, min_samples_step=1):
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
        - eps_min (float): 
            The minimum value of the eps parameter to try during optimization. Default is 0.1.
        - eps_max (float): 
            The maximum value of the eps parameter to try during optimization. Default is 1.0.
        - eps_step (float): 
            The step size for the eps parameter during optimization. Default is 0.1.
        - min_samples_min (int): 
            The minimum value of the min_samples parameter to try during optimization. Default is 2.
        - min_samples_max (int): 
            The maximum value of the min_samples parameter to try during optimization. Default is 10.
        - min_samples_step (int):
            The step size for the min_samples parameter during optimization. Default is 1.
    -----------------------------------------------------------------------------------
    Returns:
    -----------------------------------------------------------------------------------
        A tuple of the optimal values for eps and min_samples, 
        as determined by the silhouette score.
    -------------------------------------------------------------------------------
    """
    
    # Select the feature columns
    X = data[feature_cols].values
    
    # Normalize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Try different values of eps and min_samples and evaluate them
    scores = []
    for eps in np.arange(eps_min, eps_max + eps_step, eps_step):
        for min_samples in range(min_samples_min, min_samples_max + min_samples_step, min_samples_step):
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan.fit(X)
            if len(set(dbscan.labels_)) > 1:  # Skip if only one cluster is found
                score = silhouette_score(X, dbscan.labels_)
                scores.append((eps, min_samples, score))
                print("DBSCAN with eps={}, min_samples={} - Silhouette Score: {}".format(eps, min_samples, score))
    
    # Find the optimal values for eps and min_samples
    if len(scores) > 0:
        optimal_eps, optimal_min_samples, _ = max(scores, key=lambda x: x[2])
        print("Optimal values - eps: {}, min_samples: {}".format(optimal_eps, optimal_min_samples))
        return (optimal_eps, optimal_min_samples)
    else:
        print("Could not find optimal values.")
        return None

  
if __name__ == "__main__":  
    
   #Example of application
    np.random.seed(123)
    data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100)
    })

    optimal_params = train_and_optimize_DBSCAN(data, ['feature1', 'feature2', 'feature3'])
