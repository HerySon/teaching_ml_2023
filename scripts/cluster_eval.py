import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score as ss
import data_loader



def elbow_method(kmeans, data, max_clusters=10):
    """
    Applies the elbow method to determine the optimal number of clusters for KMeans.

    Args:
    - data: numpy array or pandas dataframe
    - max_clusters: int, maximum number of clusters to test

    Returns:
    - None. Plots the Within-Cluster-Sum-of-Squares (WCSS) for each number of clusters

        WCSS :
    The goal of clustering is to minimize the WCSS, since a smaller value means that the data points in the cluster
    are closer to the "center" of the cluster. The process is similar to the minimizing of the loss function.
    """
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans.n_clusters = i
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, max_clusters + 1), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.show()

def silhouette_analysis(kmeans, data, max_clusters=10):
    """
    Applies the Silhouette analysis to determine the optimal number of clusters for KMeans.

    Args:
    - data: numpy array or pandas dataframe
    - max_clusters: int, maximum number of clusters to test

    Returns:
    - None. Plots the Silhouette score for each number of clusters

        Silhouette score:
    To calculate the overall silhouette score, we first calculate the silhouette score for each data points, depending how
    well the data point is matched to it's cluster, a score between -1 and +1 is attributed, and then the average for all data
    points of the dataset is calculated, giving us the overall silhouette score.
        Analysis:
    Above 0.7 is considered to be a strong indication of well-separated clusters
    Between 0.5 and 0.7 is considered to indicate moderate separation of clusters
    Below 0.5 is considered to be a weak indication of separation of clusters
    """
    silhouette_scores = []
    for i in range(2, max_clusters + 1):
        kmeans.n_clusters = i
        kmeans.fit(data)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))
    plt.plot(range(2, max_clusters + 1), silhouette_scores)
    plt.title('Silhouette Analysis')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()

def silhouette_score(data, labels):
    """
    Calculates the Silhouette score for a given clustering.

    Args:
    - data: numpy array or pandas dataframe
    - labels: array-like, predicted cluster labels for each data point

    Returns:
    - float, the Silhouette score for the given clustering
    """
    return ss(data, labels)

def plot_radar_chart(ax, data, column_names, cluster_number, color):
    num_vars = len(column_names)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    data = np.concatenate((data, [data[0]]))
    angles += angles[:1]

    ax.plot(angles, data, color=color, linewidth=2, linestyle='solid', label=f'Cluster {cluster_number}')
    ax.fill(angles, data, color=color, alpha=0.25)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(column_names)

def plot_all_clusters(cluster_centroids, column_names, title='Radar Chart - All Clusters'):
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']

    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(polar=True))
    for i, centroid in enumerate(cluster_centroids):
        plot_radar_chart(ax, centroid, column_names, i + 1, colors[i % len(colors)])

    ax.set_title(title, size=20, color='black', y=1.05)
    ax.legend(loc='upper right', bbox_to_anchor=(1.5, 1))
    plt.show()


def test():
    df = data_loader.get_data(file_path="../data/en.openfoodfacts.org.products.csv", nrows=500)
    # Preprocess the data: choose relevant columns and handle missing values
    selected_columns = ['energy_100g', 'fat_100g', 'carbohydrates_100g', 'proteins_100g', 'fiber_100g']
    df = df[selected_columns]

    # Fill missing values with the mean of the corresponding column
    for column in selected_columns:
        df[column].fillna(df[column].mean(), inplace=True)

    # Scale the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)

    kmeans = KMeans(n_clusters=3, n_init=10, random_state=0)

    # Apply the elbow method
    elbow_method(kmeans, data_scaled, 30)

    # Apply the Silhouette analysis
    silhouette_analysis(kmeans, data_scaled, 30)

    kmeans.n_clusters = 3
    kmeans.fit(data_scaled)
    cluster_centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    score = silhouette_score(data_scaled, labels)
    print("Silhouette score:", score)
    # Plot radar chart for all cluster centroids
    plot_all_clusters(cluster_centroids, selected_columns)

test()
