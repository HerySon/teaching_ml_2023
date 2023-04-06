def dbscan(dataset, inp_eps=0.5, inp_algo='auto', inp_metric='euclidean'):
    """
     This function will fit a DBSCAN model. You pass the OpenFoodFact dataset as entry and you get the fitted model at the end.
    You can precise the number of cluster expected.

    Args :
        dataset -- the dataset who contains OpenFoodFact data (required)
        "algo":  The algorithm to be used by the NearestNeighbors module to compute pointwise distances and find nearest neighbors. Possible values are ['brute', 'ball_tree', 'kd_tree', 'auto'] -- Default = 'auto'
        "metric": The metric to use when calculating distance between instances in a feature array. Possible values are ["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan"] -- Default = 'euclidean'
        "eps": The maximum distance between two samples for one to be considered as in the neighborhood of the other. (float) -- Default = 0.5

    Returns :
        Fitted model of DBSCAN.
    """
    import pandas as pd
    import numpy as np
    from numpy.testing import assert_equal
    from sklearn.cluster import DBSCAN

    assert_equal(type(dataset), type(pd.DataFrame()), err_msg='Input is not Pandas Dataframe.', verbose=True)
    for i in dataset.columns:
        assert dataset[i].dtype in [type(5), type(5.5), np.int64().dtype, np.int32().dtype, np.float64().dtype,
                                    np.float32().dtype], f'{i} column of the dataset is not numeric type. Please ' \
                                                         f'convert columns to numeric type or input only a part of ' \
                                                         f'the dataset with numeric columns only.'
    assert_equal(dataset.isna().sum().sum(), 0,
                 err_msg=f'There is {dataset.isna().sum().sum()} NaN values in dataset, please preprocess them before '
                         f'trying to fit DBSCAN.')
    model = DBSCAN(algorithm=inp_algo, eps=inp_eps, metric=inp_metric)
    assert_equal(type(model), type(DBSCAN()), err_msg='Output will not be DBSCAN class instance.', verbose=True)
    return model
