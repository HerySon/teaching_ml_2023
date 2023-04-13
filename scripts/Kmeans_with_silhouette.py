from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns

def kmeans_with_silhouette(dataset,n):
    # Extract the numeric columns from the dataset
    numeric_cols = dataset.select_dtypes(include='number')
    
    # Try different numbers of clusters and choose the one with the highest silhouette score
    best_num_clusters = 0
    best_silhouette_score = -1
    for num_clusters in range(2, n):
        kmeans = KMeans(n_clusters=num_clusters)
        labels = kmeans.fit_predict(numeric_cols)
        silhouette_avg = silhouette_score(numeric_cols, labels)
        if silhouette_avg > best_silhouette_score:
            best_num_clusters = num_clusters
            best_silhouette_score = silhouette_avg
            
    # Fit the KMeans model with the best number of clusters
    kmeans = KMeans(n_clusters=best_num_clusters)
    labels = kmeans.fit_predict(numeric_cols)
    
    # Add the cluster labels to the original dataset
    dataset['cluster'] = labels
    
    # Plot the clusters using Seaborn
    sns.pairplot(dataset, hue='cluster')
    
    return dataset
