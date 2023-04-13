####################################
# Reduction de variables par l'UMAP#
####################################
# Import library
import numpy as np
import pandas as pd
import umap.umap_ as umap
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class UMAPTransformer(BaseEstimator, TransformerMixin):
    """
    Class to transform datas by using the UMAP algorithme.
    """
    def __init__(self, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_components = n_components
        self.metric = metric
        
    def fit(self, X, y=None):
        """
        Train the UMAP algorithme on X datas, Here our df dataset.
        """
        self.umap_model_ = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            n_components=self.n_components,
            metric=self.metric
        )
        scaler = StandardScaler()
        self.X_scaled_ = scaler.fit_transform(X)
        self.umap_model_.fit(self.X_scaled_)
        return self
    
    def transform(self, X):
        """
        Reduce the dimensions of X by using the trained UMAP algorithm.
        """
        X_scaled = StandardScaler().fit_transform(X)
        umap_result = self.umap_model_.transform(X_scaled)
        return umap_result


"""
We can now use this UMAPTransformer class as a transformer in a scikit-learn pipeline 
to reduce the dimensions of our df dataset. For exemple
"""
# Import library 

from sklearn.pipeline import Pipeline


# Create a pipeline with UMAPTransformer and a classifier
pipeline = Pipeline([
    ('umap', UMAPTransformer()),
    ('classifier', RandomForestClassifier())
])

# Divide the dataset in train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['target']), df['target'], test_size=0.2, random_state=42)

# Train and evaluate the model
pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)
print(f"Accuracy : {accuracy}")
