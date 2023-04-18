from sklearn.manifold import TSNE
from DimensionalityReduction import DimensionalityReduction


class TSNEReduction(DimensionalityReduction):
    """
    Class for t-Distributed Stochastic Neighbor Embedding (t-SNE) dimensionality reduction.
    """

    def __init__(self, data, n_components=2, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self.n_components = n_components

    def reduce_dimensionality(self):
        """
        Reduce the dimensionality of the input data using t-SNE.

        Returns:
            np.array: The reduced data as a NumPy array.
        """
        preprocessed_data = self.preprocess_data(self.data)
        tsne = TSNE(n_components=self.n_components)
        reduced_data = tsne.fit_transform(preprocessed_data)
        return reduced_data

    def fit_transform(self):
        """
        Apply the t-SNE dimensionality reduction method and return the reduced data.

        Returns:
            np.array: The reduced data as a NumPy array.
        """
        tsne = TSNE(n_components=2, perplexity=5)
        reduced_data = tsne.fit_transform(self.data)
        return reduced_data
