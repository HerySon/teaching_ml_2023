from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import numpy as np

def incremental_kmeans_with_silhouette(dataset, n_clusters, batch_size, init_size=None):
    """
    Perform incremental KMeans clustering on a dataset and return the dataset with cluster labels.
    
    Parameters:
    -----------
    dataset : pandas DataFrame
        The input dataset.
    n_clusters : int The number of clusters to form.
    batch_size : int, The batch size
        The size of the mini-batches.
    init_size : int, optional (default=None)
        Number of samples to randomly initialize KMeans at the beginning.
        
    Returns:
    --------
    dataset : pandas DataFrame
        The input dataset with an additional 'cluster' column containing cluster labels.
    """
    # Extract the numeric columns from the dataset
    numeric_cols = dataset.select_dtypes(include='number')

    # Initialize the MiniBatchKMeans algorithm
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, init_size=init_size, random_state=42)

    # Initialize the silhouette scores list
    silhouette_scores = []

    # Initialize the number of clusters to 2
    num_clusters = 2

    # Loop until the maximum number of clusters is reached
    while num_clusters <= n_clusters:
        # Fit the KMeans model
        kmeans.fit(numeric_cols)

        # Predict the cluster labels for each sample
        labels = kmeans.predict(numeric_cols)

        # Compute the silhouette score for the current number of clusters
        silhouette_avg = silhouette_score(numeric_cols, labels)

        # Append the silhouette score to the list
        silhouette_scores.append(silhouette_avg)

        # Increment the number of clusters
        num_clusters += 1

    # Find the optimal number of clusters with the highest silhouette score
    best_num_clusters = np.argmax(silhouette_scores) + 2  # +2 because we started from 2 clusters

    # Fit the KMeans model with the optimal number of clusters
    kmeans = MiniBatchKMeans(n_clusters=best_num_clusters, batch_size=batch_size, init_size=init_size, random_state=42)
    kmeans.fit(numeric_cols)

    # Predict the cluster labels for each sample
    labels = kmeans.predict(numeric_cols)

    # Add the cluster labels to the original dataset
    dataset['cluster'] = labels
    
    #Print the best number of clusters
    print('The best number of clusters within the given range is :',best_num_clusters)
    
    return dataset
