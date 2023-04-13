import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def train_and_optimize_kmeans(data, feature_cols, num_clusters=3, max_clusters=10):
    """
    Trains and optimizes K-means on the OpenFoodFacts dataset.
    
    Args:
        data_path (str): The path to the OpenFoodFacts dataset file.
        feature_cols (list): A list of column names for the features to be used for training.
        num_clusters (int): The number of clusters to use for the initial K-means training. Default is 3.
        max_clusters (int): The maximum number of clusters to try during the optimization. Default is 10.
    
    Returns:
        The optimal number of clusters, as determined by the silhouette score.
    """
    
    
    # Select the feature columns
    X = data[feature_cols].values
    
    # Apply dimensionality reduction if needed
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    
    # Normalize the data if needed
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Train the initial K-means model
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    
    # Evaluate the initial model
    inertia = kmeans.inertia_
    silhouette = silhouette_score(X, kmeans.labels_)
    print("Initial model - Inertia: {}, Silhouette Score: {}".format(inertia, silhouette))
    
    # Try different numbers of clusters and evaluate them
    scores = []
    for i in range(2, max_clusters+1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        scores.append(score)
        print("K-means with {} clusters - Silhouette Score: {}".format(i, score))
    
    # Find the optimal number of clusters
    optimal_clusters = np.argmax(scores) + 2
    print("Optimal number of clusters: {}".format(optimal_clusters))
    
    return optimal_clusters
  
if __name__ == "__main__":  
    
    # Example usage
    data = pd.DataFrame({'col1': [1, 2, 3, 4, 5], 'col2': [6, 7, 8, 9, 10], 'col3': [11, 12, 13, 14, 15]})
    feature_cols = ['col1', 'col2', 'col3']
    num_clusters = 3
    max_clusters = 10

    optimal_clusters = train_and_optimize_kmeans(data, feature_cols, num_clusters, max_clusters)
