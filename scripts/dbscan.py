from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

def dbscan_clustering(data, eps=0.5, min_samples=5):
    """
    Applies DBSCAN clustering to the input data and returns the cluster labels and the silhouette score.
    
    Args:
    data: pandas DataFrame or numpy array of shape (n_samples, n_features)
        The input data to cluster.
    eps: float, optional (default=0.5)
        The maximum distance between two samples for them to be considered as in the same neighborhood.
    min_samples: int, optional (default=5)
        The number of samples in a neighborhood for a point to be considered as a core point.
        
    Returns:
    labels: numpy array of shape (n_samples,)
        The cluster labels assigned by DBSCAN. Outliers are assigned -1.
    silhouette: float
        The silhouette score of the clustering.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    
    # Calculate silhouette score, ignoring outliers (-1)
    if -1 in labels:
        silhouette = silhouette_score(data[labels!=-1], labels[labels!=-1])
    else:
        silhouette = silhouette_score(data, labels)
    
    return labels, silhouette
