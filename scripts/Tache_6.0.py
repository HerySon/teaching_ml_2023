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
    In this example I train the AgglomerativeClustering algorithm on 3 precise columns :
    - energy_100g
    - fat_100g
    - saturated-fat_100g
    By executing the function below two graphs will be saved :
    - Elbow_agg.png : Shows the elbow to have a visual of the ideal cluster number to give to the algorithm
    - Clusters_agg.png : Shows different clusters
"""

def algo_hierarchical_cluster(dataset):
    """Clean and preprocess data"""
    dataset = dataset[(dataset['energy_100g'] > 0) & (dataset['fat_100g'] > 0) & (dataset['saturated-fat_100g'] > 0)]

    """Divide the data into two sets: a training set and a test set"""
    df_train = dataset.sample(frac=0.8, random_state=42)
    df_test = dataset.drop(df_train.index)

    """Normalize training data"""
    scaler = StandardScaler()
    scaler_train = scaler.fit_transform(df_train[['energy_100g', 'fat_100g', 'saturated-fat_100g']])

    """Training AgglomerativeClustering algorithm"""
    agg = AgglomerativeClustering(n_clusters=3)
    agg.fit(scaler_train)

    """Calculate silhouette measurement for Agglomerative Clustering"""
    agg_silhouette = silhouette_score(scaler_train, agg.labels_)

    """Plot the sum of squares of the intra-luster distances as a function of the number of clusters"""
    inert = []
    for k in range(1, 10):
        agg = AgglomerativeClustering(n_clusters=k, random_state=42)
        agg.fit(scaler_train)
        inert.append(agg.inertia_)

    plt.plot(range(1, 10), inert, 'bx-')
    plt.xlabel('Nombre de clusters')
    plt.savefig('Elbow_agg.png')
    plt.show()

    """Draw a scatterplot of the clusters"""
    plt.scatter(scaler_train[:, 0], scaler_train[:, 1], c=agg.labels_)
    plt.title(f'Clusters de l\'algorithme de clustering hiérarchique agg - Score: {agg_silhouette}')
    plt.xlabel('Energie pour 100g')
    plt.ylabel('Matières grasses pour 100g')
    plt.savefig('Cluster_agg.png')
        
if __name__ == "__main__":
    dataset_directory = "data\en.openfoodfacts.org.products.csv"
    displayed_rows = 100
    df = pd.read_csv(dataset_directory, nrows = displayed_rows, sep='\t', encoding='utf-8')
    algo_hierarchical_cluster(df)