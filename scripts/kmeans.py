def kmeans(dataset, k=8, algorithm='lloyd'):
    """
     This function will fit a KMeans model. You pass the OpenFoodFact dataset as entry and you get the fitted model at the end.
    You can precise the number of cluster and the algorithm used to define the clusters.

    Args :
        dataset -- the dataset who contains OpenFoodFact data (required)
        k -- number of clusters -- Default = 8
        algorithm -- type of algorithm used. Possibility are : ['lloyd', 'elkan', 'auto', 'full'] (refers to https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html for more informations) -- Default = 'lloyd'

    Returns :
        Fitted model of KMeans.
    """
    import pandas as pd
    import numpy as np
    from numpy.testing import assert_equal
    from sklearn.cluster import KMeans
    assert_equal(type(dataset), type(pd.DataFrame()), err_msg='Input is not Pandas Dataframe.', verbose=True)
    model = KMeans(n_clusters=k, random_state=13, algorithm=algorithm)
    model.fit(dataset)
    assert_equal(type(model), type(KMeans()), err_msg='Output will not be KMeans class instance.', verbose=True)
    return model
