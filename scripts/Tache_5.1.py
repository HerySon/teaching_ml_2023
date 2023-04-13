import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

""" 
Args :
    dataset = Dataframe of your dataset
Operating :
    In this function we will drive three clustering algorithms :
    - K-Means
    - DBSCAN
    - AgglomerativeClustering
    By executing the function below one graph will be saved and the three silhouette score of each algorithm will be printed:
    - Comparaison_algo_clustering.png : Shows cluster detection of two algorithms (K-Means and DBSCAN)
    - kmeans_silhouette,dbscan_silhouette and agg_silhouette : print silhouette score
"""

def compare_algo_cluster(dataset):
    """Clean and preprocess data"""
    dataset = dataset[(dataset['energy_100g'] > 0) & (dataset['fat_100g'] > 0) & (dataset['saturated-fat_100g'] > 0)]

    """Divide the data into two sets: a training set and a test set"""
    df_train = dataset.sample(frac=0.8, random_state=42)
    df_test = dataset.drop(df_train.index)

    """Normalize training data"""
    scaler = StandardScaler()
    scaler_train = scaler.fit_transform(df_train[['energy_100g', 'fat_100g', 'saturated-fat_100g']])

    """Training DBSCAN algorithm"""
    dbscan = DBSCAN(eps=0.3)
    dbscan.fit(scaler_train)

    """Training AgglomerativeClustering algorithm"""
    agg = AgglomerativeClustering(n_clusters=3)
    agg.fit(scaler_train)

    """Training Kmeans algorithm"""
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(scaler_train)

    """Calculate silhouette measurement for K-means"""
    kmeans_silhouette = silhouette_score(scaler_train, kmeans.labels_)
    print(f"Mesure de silhouette pour K-means : {kmeans_silhouette}")

    """Calculate silhouette measurement for DBSCAN"""
    dbscan_silhouette = silhouette_score(scaler_train, dbscan.labels_)
    print(f"Mesure de silhouette pour DBSCAN : {dbscan_silhouette}")

    """Calculate silhouette measurement for Agglomerative Clustering"""
    agg_silhouette = silhouette_score(scaler_train, agg.labels_)
    print(f"Mesure de silhouette pour Agglomerative Clustering : {agg_silhouette}")

    """Plot a scatterplot of clusters for the K-means algorithm"""
    plt.subplot(1, 2, 1)
    plt.scatter(scaler_train[:, 0], scaler_train[:, 1], c=kmeans.labels_)
    plt.title('Clusters de l\'algorithme K-means')
    plt.xlabel('Energie pour 100g')
    plt.ylabel('Matières grasses pour 100g')

    """Plot a scatterplot of clusters for the Agglomerative Clustering algorithm"""
    plt.subplot(1, 2, 2)
    plt.scatter(scaler_train[:, 0], scaler_train[:, 1], c=dbscan.labels_)
    plt.title('Clusters de l\'algorithme Agglomerative Clustering')
    plt.xlabel('Energie pour 100g')
    plt.ylabel('Matières grasses pour 100g')
    plt.subplots_adjust(wspace=2)

    """Save the chart to a file"""
    plt.savefig('Comparaison_algo_clustering.png')
    plt.show()
        
if __name__ == "__main__":
    dataset_directory = "data\en.openfoodfacts.org.products.csv"
    displayed_rows = 100
    df = pd.read_csv(dataset_directory, nrows = displayed_rows, sep='\t', encoding='utf-8')
    compare_algo_cluster(df)