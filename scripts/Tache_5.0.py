import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

""" 
Args :
    dataset = Dataframe of your dataset
    nb_cluster = Number of clusters the user want. 
    The value can be changed according to new observations
Operating :
    In this example I train the Kmeans algorithm on 3 precise columns :
    - energy_100g
    - fat_100g
    - saturated-fat_100g
    By executing the function below two graphs will be saved :
    - Elbow_Kmeans.png : Shows the elbow to have a visual of the ideal cluster number to give to the algorithm
    - Clusters_Kmeans.png : Shows different clusters
"""

def algo_Kmeans(dataset, nb_cluster):
    """Clean and preprocess data"""
    dataset = dataset[(dataset['energy_100g'] > 0) & (dataset['fat_100g'] > 0) & (dataset['saturated-fat_100g'] > 0)]
    """Divide the data into two sets: a training set and a test set"""
    df_train = dataset.sample(frac=0.8, random_state=42)
    df_test = dataset.drop(df_train.index)
    """Normalize the data"""
    scaler = StandardScaler()
    scaler_train = scaler.fit_transform(df_train[['energy_100g', 'fat_100g', 'saturated-fat_100g']])
    """Training Kmeans algorithm"""
    kmeans = KMeans(n_clusters=nb_cluster, random_state=42)
    kmeans.fit(scaler_train)
    """Evaluate the performance of the K-means algorithm using the test set"""
    scaler_test = scaler.transform(df_test[['energy_100g', 'fat_100g', 'saturated-fat_100g']])
    predictions = kmeans.predict(scaler_test)
    silhouette_avg = silhouette_score(scaler_test, predictions)

    """Plot the sum of squares of the intra-luster distances as a function of the number of clusters"""
    inert = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaler_train)
        inert.append(kmeans.inertia_)

    plt.plot(range(1, 10), inert, 'bx-')
    plt.xlabel('Nombre de clusters')
    plt.savefig('Elbow_Kmeans.png')
    plt.show()
    """Draw a scatterplot of the clusters"""
    plt.scatter(scaler_train[:, 0], scaler_train[:, 1], c=kmeans.labels_)
    plt.title(f'Clusters de l\'algorithme K-means - Score: {silhouette_avg}')
    plt.xlabel('Energie pour 100g')
    plt.ylabel('Matières grasses pour 100g')
    plt.savefig('Clusters_Kmeans.png')
    plt.show()
        
if __name__ == "__main__":
    dataset_directory = "data\en.openfoodfacts.org.products.csv"
    displayed_rows = 100
    df = pd.read_csv(dataset_directory, nrows = displayed_rows, sep='\t', encoding='utf-8')
    algo_Kmeans(df,4)