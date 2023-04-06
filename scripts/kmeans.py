def kmeans(dataset, k=8):
    """
     This function will fit a KMeans model. You pass the OpenFoodFact dataset as entry and you get the fitted model at the end.
    You can precise the number of cluster expected.

    Args :
        dataset -- the dataset who contains OpenFoodFact data (required)
        k -- number of clusters -- Default = 8

    Returns :
        Fitted model of KMeans.
    """
    import pandas as pd
    import numpy as np
    from numpy.testing import assert_equal
    from sklearn.cluster import KMeans
    from sklearn.model_selection import GridSearchCV
    assert_equal(type(dataset), type(pd.DataFrame()), err_msg='Input is not Pandas Dataframe.', verbose=True)
    for i in dataset.columns:
        assert dataset[i].dtype in [type(5), type(5.5), np.int64().dtype, np.int32().dtype, np.float64().dtype,
                                    np.float32().dtype], f'{i} column of the dataset is not numeric type. Please ' \
                                                         f'convert columns to numeric type or input only a part of ' \
                                                         f'the dataset with numeric columns only.'
    assert_equal(dataset.isna().sum().sum(), 0,
                 err_msg=f'There is {dataset.isna().sum().sum()} NaN values in dataset, please preprocess them before '
                         f'trying to fit KMeans.')
    model = KMeans(n_clusters=k, random_state=13, n_init=10)
    gs_cv = GridSearchCV(model, cv=5, param_grid=[{"algorithm": ['lloyd', 'elkan', 'auto', 'full']},
                                                  {"init": ["k-means++", "random"]},
                                                  {"n_init": ["auto", 5, 10, 25, 50]}])
    gs_cv.fit(dataset)
    model = gs_cv.best_estimator_
    assert_equal(type(model), type(KMeans()), err_msg='Output will not be KMeans class instance.', verbose=True)
    return model
