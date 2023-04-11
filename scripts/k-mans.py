def __init__(data, n_clusters, max_iter=100, random_state=123):  # initialisation du cluster 
       # cette fonction permet de charger un dataset pour entrainer un model de Kmans 
       # specifier les arguments :

        data.n_clusters = n_clusters 
        data.max_iter = max_iter
        data.random_state = random_state

        # n_clusters : nombre de cluster 
         # max_iter : valeur maximal iterration 
         # random_state : initialisation d'une valeur possible aleatoire 
 # Initialisons les centroïdes en mélangeant d’abord le jeu de données
  def initializ_centroids(data, X):
        np.random.RandomState(data.random_state)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:data.n_clusters]]
        return centroids
        
# itérer jusqu’à ce qu’il n’y ait plus de changement aux centroïdes.
# Calculez la somme de la distance au carré entre les points de données et tous les centroïdes
 def compute_distance(data, X, centroids):
        distance = np.zeros((X.shape[0], data.n_clusters))
        for k in range(data.n_clusters):
            row_norm = norm(X - centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)
        return distance
def find_closest_cluster(data, distance):
        return np.argmin(distance, axis=1)
#Affectez chaque point de données au cluster le plus proche (centroïde)
 def compute_sse(data, X, labels, centroids):
        distance = np.zeros(X.shape[0])
        for k in range(data.n_clusters):
            distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1)
        return np.sum(np.square(distance))
#alculez les centroïdes pour les clusters en prenant la moyenne de tous les points de données appartenant à chaque cluster.
def fit(data, X):
        data.centroids = data.initializ_centroids(X)
        for i in range(data.max_iter):
            old_centroids = data.centroids
            distance = data.compute_distance(X, old_centroids)
            data.labels = data.find_closest_cluster(distance)
            data.centroids = data.compute_centroids(X, data.labels)
            if np.all(old_centroids == data.centroids):
                break
        data.error = data.compute_sse(X, data.labels, data.centroids)