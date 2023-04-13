import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def kmeans(dataset, k_min=2, k_max=10, method='elbow', inp_algo='auto', inp_init='k-means++', inp_n_init=10):
    """
    This function will fit a KMeans model. You pass the OpenFoodFact dataset as entry and you get the fitted model at the end.
    You can precise the number of cluster expected.
    Args :
        dataset -- the dataset who contains OpenFoodFact data (required)
        
        k_min -- number of cluster minimum -- Default = 2 (int)
        
        k_max -- number of cluster maximum -- Default = 10 (int)
        
        method -- Method for selecting the number of clusters. Possible values are ['elbow', 'silhouette'] -- Default = 'elbow'
        
        inp_algo -- K-means algorithm to use. Possible values are ['auto', 'full', 'elkan'] -- Default = 'auto'
        
        inp_init -- Method for initialization of centroids. Possible values are ["k-means++", "random"] -- Default = 'k-means++'
        
        inp_n_init -- Number of times the k-means algorithm will be run with different centroid seeds. 
        Possible values are "auto" or an integer. -- Default = 10 (int)
    Returns :
        Fitted model of KMeans.
    """
    
    # Verifying if dataset input is a pd.DataFrame().
    assert isinstance(dataset, pd.DataFrame), 'Input is not a Pandas DataFrame.'
    
    # Verifying if the dataset contains only numerical columns.
    assert all(dataset.dtypes.apply(lambda x: np.issubdtype(x, np.number))), \
    'The dataset contains non-numerical columns. Please select only numerical columns or convert them to numerical data types.'
    
    # Verifying if there are any missing values in the dataset.
    assert dataset.isnull().sum().sum() == 0, 'The dataset contains missing values. Please preprocess the dataset before fitting KMeans.'
    
    # Creating an empty list to store the distortion values for different values of k.
    distortions = []
    
    # Creating a list of values of k to test.
    k_values = range(k_min, k_max+1)

    # Looping over different values of k to calculate the distortion values.
    for k in k_values:
        model = KMeans(n_clusters=k, algorithm=inp_algo, init=inp_init, n_init=inp_n_init)
        model.fit(dataset)
        distortions.append(model.inertia_)
    
    # Plotting the elbow curve.
    if method == 'elbow':
        plt.plot(k_values, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')
        plt.show()
        
        # Prompting the user to enter the value of k.
        k = input("Enter the value of k: ")
        
    # Using silhouette method to select k.
    elif method == 'silhouette':
        from sklearn.metrics import silhouette_score
        
        silhouette_scores = []
        for k in k_values:
            model = KMeans(n_clusters=k, algorithm=inp_algo, init=inp_init, n_init=inp_n_init)
            model.fit(dataset)
            silhouette_scores.append(silhouette_score(dataset, model.labels_))
        
        # Plotting the silhouette scores.
        plt.plot(k_values, silhouette_scores, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Silhouette Score')

def kmeans_elbow(X, k_min=2, k_max=10):
    """
    This function uses the elbow method to determine the optimal number of clusters for KMeans.

    Parameters:
    X (numpy.ndarray): The dataset to cluster
    k_min (int): The minimum number of clusters to test
    k_max (int): The maximum number of clusters to test

    Returns:
    list: The inertias for each tested number of clusters
    """

    inertias = []
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    return inertias
