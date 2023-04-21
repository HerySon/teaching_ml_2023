import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from hierarchical_clustering import hie_clustering

# Generate some fake data
X, y_true = make_blobs(n_samples=100, centers=3, cluster_std=1.0, random_state=42)

# Convert the data to a pandas DataFrame
df = pd.DataFrame(X, columns=["x", "y"])

# Perform hierarchical clustering and plot the dendrogram
model, labels = hie_clustering(df, n_clusters=None, plotdendrogram=True)

# Plot the clustering results
plt.scatter(df["x"], df["y"], c=labels, cmap="viridis")
plt.title("Hierarchical Clustering Results")
plt.show()
