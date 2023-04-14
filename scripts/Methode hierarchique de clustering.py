import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

def hierarchical_clustering(data, n_clusters=None, linkage='ward', distance_threshold=None, plot_dendrogram=False):
    """
    Performs hierarchical clustering on the input data.

    Parameters:
    -----------
    data : pandas.DataFrame or numpy.ndarray
        The input data to cluster.
    n_clusters : int, optional
        The number of clusters to form, default is None.
    linkage : str, optional
        The linkage criterion to use. Can be one of "ward", "complete", "average", or "single". Default is "ward".
    distance_threshold : float, optional
        The distance threshold to use for forming flat clusters. If this is not None, then n_clusters must be None.
    plot_dendrogram : bool, optional
        Whether to plot the dendrogram of the hierarchical clustering. Default is False.

    Returns:
    --------
    clusters : numpy.ndarray
        An array of length n_samples giving the cluster labels for each sample.
    """
    # Check input data type and convert to numpy array if necessary
    if isinstance(data, pd.DataFrame):
        data = data.values
    elif not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a pandas DataFrame or numpy ndarray.")

    # Create hierarchical clustering model and fit to data
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, distance_threshold=distance_threshold)
    model.fit(data)

    # Plot dendrogram if desired
    if plot_dendrogram:
        dendrogram(model.children_, labels=model.labels_)

    # Return cluster labels
    return model.labels_
