import numpy as np
import math
import matplotlib.pylab as plt
from scipy.spatial.distance import pdist, squareform


class KMeanPrecomputed:
    def __init__(data, distance_matrix):
        data.distances = distance_matrix
        data.point_count = distance_matrix.shape[0]

        data.nearest_centroids = None
        data.centroids = None
        data.cluster_count = 0
        data.cost = 0

    def run(data, cluster_count=2, max_iterations=1000):
        data.cluster_count = cluster_count

        data.clusters_init()
        last_cost = 0
        data.clusters_match()
        for i in range(max_iterations):
            data.clusters_move()
            data.clusters_match()

            data.cost = data.cluster_cost()

            if last_cost == data.cost:
                break
            last_cost = data.cost

        return data.nearest_centroids

    def clusters_init(data):
        data.centroids = np.random.choice(data.point_count, data.cluster_count, replace=False)
        data.nearest_centroids = np.random.choice(data.cluster_count, data.point_count)

    def clusters_match(data):
        data.nearest_centroids = data.distances[:, data.centroids].argmin(axis=1)

    def clusters_move(data):
        for i, centroid in enumerate(data.centroids):
            points_in_cluster = data.nearest_centroids == i
            new_centroid_index = data.distances[points_in_cluster][:, points_in_cluster].sum(axis=0).argmin()
            new_centroid = np.argwhere(points_in_cluster.cumsum() == new_centroid_index + 1)[0][0]
            data.centroids[i] = new_centroid

    def cluster_cost(data):
        return data.distances[range(data.point_count), data.centroids[data.nearest_centroids]].sum() / data.point_count


def demo():
    dat = 1000
    a = np.empty((4*dat,2),float)
    for i in range(dat):
        a[i] = np.random.multivariate_normal([0,0],[[1,0.1],[1,.7]])
        a[i+dat] = np.random.multivariate_normal([3,2],[[0.5,0],[0,2]])
        a[i+2*dat] = np.random.multivariate_normal([3,2],[[0.5,0],[0,2]])
        a[i+3*dat] = np.random.multivariate_normal([0,4],[[0.1,0],[0,.1]])

    distance_matrix = squareform(pdist(a, 'euclidean'))
    km = KMeanPrecomputed(distance_matrix)
    labels = km.run(3)
    z = (labels == 2)
    x = (labels == 1)
    y = (labels == 0)


    plt.plot(a[x,0],a[x,1],"ro")
    plt.plot(a[y,0],a[y,1],"bo")
    plt.plot(a[z, 0],a[z, 1],"go")
    plt.show()