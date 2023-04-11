def kmeans(dataset, k_min=2, k_max=10, elbow_choice='inertias', inp_algo='lloyd', inp_init='k-means++', inp_n_init=10):
    """
     This function will fit a KMeans model. You pass the OpenFoodFact dataset as entry and you get the fitted model at the end.
    You can precise the number of cluster expected.

    Args :
        dataset -- the dataset who contains OpenFoodFact data (required)
        
        k_min -- number of cluster minimum -- Default = 2 (int)
        
        k_max -- number of cluster maximum -- Default = 10 (int)
        
        elbow_choice -- whether to use inertias or distortions for elbow method.
        Possible values are 'inertias' and 'distortions'. -- Default = 'inertias'
        
        inp_algo -- K-means algorithm to use. Possible values are ['lloyd', 'elkan', 'auto', 'full'] -- Default = 'lloyd'
        
        inp_init -- Method for initialization of centroids. Possible values are ["k-means++", "random"] -- Default = 'k-means++'
        
        inp_n_init -- Number of times the k-means algorithm is run with different centroid seeds. 
        Possible values are "auto" or an integer. -- Default = 10 (int)

    Returns :
        Fitted model of KMeans.
    """
    import pandas as pd
    import numpy as np
    from numpy.testing import assert_equal
    from sklearn.cluster import KMeans
    
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
                         f'trying to fit KMeans.')
    
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    K = range(k_min, k_max)

    for z in K:
        # Building and fitting the model
        kmeanModel = KMeans(n_clusters=z).fit(X)
        kmeanModel.fit(X)

        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                            'euclidean'), axis=1)) / X.shape[0])
        inertias.append(kmeanModel.inertia_)

        mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                       'euclidean'), axis=1)) / X.shape[0]
        mapping2[k] = kmeanModel.inertia_
    
    if elbow_choice == "inertias":
        choice = inertias
    elif elbow_choice == "distortions":
        choice = distortions
    plt.plot(K, choice, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Inertia')
    plt.title('The Elbow Method using Inertia')
    plt.show()
    
    k = input("Now choose a value for k :")
    
    model = KMeans(n_clusters=k, random_state=13, algorithm=inp_algo, init=inp_init, n_init=inp_n_init)
    return model.fit(dataset)
