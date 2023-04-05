import pandas as pd
import numpy as np
import logging
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration

def hie_clustering(df,modeltype=AgglomerativeClustering,n_clusters=2,linkage='ward',metric='euclidean'):
    """
    Hierachical clustering method from Scikit-Learn.
    Args:
        df : pandas dataframe (The input dataframe)
        modeltype : which clustering algorithm to use (default=AgglomerativeClustering)
            'AgglomerativeClustering' : performs a hierarchical clustering using a bottom up approach: each observation starts in its own cluster, and clusters are successively merged together.
            'FeatureAgglomeration' : uses agglomerative clustering to group together features that look very similar, thus decreasing the number of features.
        n_clusters : number of cluster to find (default=2)
        link : Which linkage criterion to use. The linkage criterion determines which distance to use between sets of features.
            The algorithm will merge the pairs of cluster that minimize this criterion (default='ward')
                'ward' minimizes the variance of the clusters being merged
                'average' uses the average of the distances of each observation of the two sets
                'complete' linkage uses the maximum distances between all observations of the two sets
                'single' uses the minimum of the distances between all observations of the two sets
        metric : Metric used to compute the linkage (default='euclidean')
            Can be 'euclidean', 'l1', 'l2', 'manhattan', 'cosine', or 'precomputed'.
            If linkage is 'ward', only 'euclidean' is accepted.
    Return:
        model with clusters
    """
    if linkage == 'ward' and metric != 'euclidean':
        metric='euclidean'
        logging.warning("If linkage is 'ward', only 'euclidean' is accepted. Automatically switch to 'euclidean'.")
                        
    model = modeltype(n_clusters=n_clusters,linkage=linkage,metric=metric).fit(df)
    return model, model.labels_