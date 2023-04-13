from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform

def agnes(dataset, linkage_method='ward', n_clusters=None, dist_metric='euclidean'):
    """
    This function will fit an AgglomerativeClustering model. You pass the OpenFoodFact dataset as entry and you get the fitted model at the end.
    Args :
        dataset -- the dataset who contains OpenFoodFact data (required)
        
        linkage_method -- Criterion used to compute the linkage. 
        Can be “ward”, “complete”, “average”, “single”. -- Default = 'ward' (str)
        
        n_clusters -- The number of clusters to form as well as the number of centroids to generate. -- Default = None (int)
        
        dist_metric -- The distance metric to use. Can be "euclidean", "manhattan", "chebyshev", "cosine", "precomputed" 
        or any metric from scipy.spatial.distance -- Default = 'euclidean' (str)
        
    Returns :
        Fitted model of AgglomerativeClustering.
    """
    import pandas as pd
    import numpy as np
    from numpy.testing import assert_equal
    
    # Verifying if dataset input is a pd.DataFrame().
    assert_equal(type(dataset), type(pd.DataFrame()), err_msg='Input is not Pandas Dataframe.', verbose=True)
    for i in dataset.columns:
        assert dataset[i].dtype in [type(int()), type(float()), np.int64().dtype, np.int32().dtype, np.float64().dtype,
                                    np.float32().dtype], f'{i} column of the dataset is not numeric type. Please ' \
                                                         f'convert columns to numeric type or input only a part of ' \
                                                         f'the dataset with numeric columns only.'
    # Verifying if the number of missing values in the dataset is 0. (no missing value)
    assert_equal(dataset.isna().sum().sum(), 0,
                 err_msg=f'There is {dataset.isna().sum().sum()} NaN values in dataset, please preprocess them before '
                         f'trying to fit AgglomerativeClustering.')

    # Calculating pairwise distances between points in the dataset
    dist_matrix = squareform(pdist(dataset, metric=dist_metric))

    # Fitting the model
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method, affinity='precomputed')
    model.fit(dist_matrix)

    return model
