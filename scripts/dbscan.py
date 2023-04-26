import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from numpy.testing import assert_equal

def dbscan_clustering(dataset, eps=0.5, min_samples=5):
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
    @Author: Thomas Leydet    
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(dataset)
    
    # Calculate silhouette score, ignoring outliers (-1)
    if -1 in labels:
        silhouette = silhouette_score(dataset[labels!=-1], labels[labels!=-1])
    else:
        silhouette = silhouette_score(dataset, labels)
    
    return labels, silhouette

def dbscan(dataset, inp_eps=0.5, inp_algo='auto', inp_metric='euclidean', inp_min_samp=5):
    """
     This function will fit a DBSCAN model. You pass the OpenFoodFact dataset as entry and you get the fitted model at the end.
    You can precise the number of cluster expected.

    Args :
        dataset -- dataframe reprensenting the dataset who contains OpenFoodFact data (required)
        
        inp_algo --  The algorithm to be used by the NearestNeighbors module to compute pointwise distances and find nearest neighbors. 
        Possible values are ['brute', 'ball_tree', 'kd_tree', 'auto'] -- Default = 'auto'
        
        inp_metric -- The metric to use when calculating distance between instances in a feature array. 
        Possible values are ["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan"] -- Default = 'euclidean'
        
        inp_eps -- The maximum distance between two samples for one to be considered as in the neighborhood of the other. (float) -- Default = 0.5
        
        inp_min_samp -- The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. 
        This includes the point itself. -- Default = 5 (int)

    Returns :
        Fitted model of DBSCAN.
        @Author : Grace
    """
    import pandas as pd
    import numpy as np
    from numpy.testing import assert_equal
    from sklearn.cluster import DBSCAN

    # Verifying if dataset input is a pd.DataFrame().
    assert_equal(type(dataset), type(pd.DataFrame()), err_msg='Input is not Pandas Dataframe.', verbose=True)
    for i in dataset.columns:
        assert dataset[i].dtype in [type(5), type(5.5), np.int64().dtype, np.int32().dtype, np.float64().dtype,
                                    np.float32().dtype], f'{i} column of the dataset is not numeric type. Please ' \
                                                         f'convert columns to numeric type or input only a part of ' \
                                                         f'the dataset with numeric columns only.'
    # Verifying if the number of missing values in the dataset is 0. (no missing value)
    assert_equal(dataset.isna().sum().sum(), 0,
                 err_msg=f'There is {dataset.isna().sum().sum()} NaN values in dataset, please preprocess them before '
                         f'trying to fit DBSCAN.')
    model = DBSCAN(algorithm=inp_algo, eps=inp_eps, metric=inp_metric, min_samples=inp_min_samp)
    assert_equal(type(model), type(DBSCAN()), err_msg='Output will not be DBSCAN class instance.', verbose=True)
    return model.fit(dataset)
