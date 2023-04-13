import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration
from scipy.cluster.hierarchy import dendrogram

def plot_dendrogram(model, **kwargs):
    """
    This function creates a linkage matrix from the model and uses it to plot a
    dendrogram of the clustering. The linkage matrix is created by counting the
    number of samples under each node of the dendrogram, and the scipy dendrogram
    function is used to actually plot the dendrogram. Additional keyword arguments
    can be passed to the scipy dendrogram function through **kwargs.

    Keyword arguments:
    ------------------
        model: AgglomerativeClustering or FeatureAgglomeration object
            The fitted hierarchical clustering model to visualize.
        **kwargs: optional keyword arguments
            Additional arguments to pass to the scipy dendrogram function.

    Returns:
    --------
        None
    """
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)



def hie_clustering(df, modeltype=AgglomerativeClustering, n_clusters=2, linkage='ward', metric='euclidean', plotdendrogram=False):
    """
    Hierachical clustering method from Scikit-Learn.

    Keyword arguments:
    ------------------
        df: pandas.DataFrame
            The input DataFrame.
        modeltype: type, optional (default AgglomerativeClustering)
            Which clustering algorithm to use. Possible values are:
            - AgglomerativeClustering : performs a hierarchical clustering using a bottom up approach:
            each observation starts in its own cluster, and clusters are successively merged together.
            - FeatureAgglomeration : uses agglomerative clustering to group together features that look
            very similar, thus decreasing the number of features.
        n_clusters: int, optional (default 2)
            Number of cluster to find.
        linkage: str, optional (default 'ward')
            Which linkage criterion to use. The linkage criterion determines which distance to use
            between sets of features. The algorithm will merge the pairs of cluster that minimize
            this criterion. Possible values are:
            - 'ward' minimizes the variance of the clusters being merged
            - 'average' uses the average of the distances of each observation of the two sets
            - 'complete' linkage uses the maximum distances between all observations of the two sets
            - 'single' uses the minimum of the distances between all observations of the two sets
        metric: str, optional (default 'euclidean')
            Metric used to compute the linkage.
            Can be 'euclidean', 'l1', 'l2', 'manhattan', 'cosine', or 'precomputed'.
            If linkage is 'ward', only 'euclidean' is accepted.
        plotdendrogram: bool, optional (default False)
            Will plot the dendogram.

    Returns:
    --------
        sklearn cluster object

    Author:
    -------
        Joëlle Sabourdy
    """
    if linkage == 'ward' and metric != 'euclidean':
        metric='euclidean'
        logging.warning("If linkage is 'ward', only 'euclidean' is accepted. Automatically switch to 'euclidean'.")
        
    if plotdendrogram==True and n_clusters != None:
        n_clusters=None
        dt=0
        logging.warning("Exactly one of n_clusters and distance_threshold has to be set, and the other needs to be None. Automatically switch to None.")
    else:
        dt=None
  
    model = modeltype(n_clusters=n_clusters,linkage=linkage,metric=metric,distance_threshold=dt).fit(df)

    if plotdendrogram:
        plt.title("Hierarchical Clustering Dendrogram")
        # plot the top three levels of the dendrogram
        plot_dendrogram(model, truncate_mode="level", p=3)
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        plt.show()

    return model, model.labels_



if __name__ == "__main__":
    # Consider dataset containing ramen rating
    df = pd.DataFrame({
        'brand': [3, 3, 1, 1, 2, 0],
        'style': [0, 0, 0, 1, 1, 0],
        'rating': [3, 4, 3.5, 1, 5, 2],
        'grams': [80, 80, 80, 90, 90, 80]
        })
    # Hierachical clustering and plot dendogram
    hie = hie_clustering(df, modeltype=AgglomerativeClustering, n_clusters=2, linkage='ward', metric='l1', plotdendrogram=True)
    print(hie)