from sklearn.manifold import Isomap
from DimensionalityReduction import DimensionalityReduction


class IsomapReduction(DimensionalityReduction):
    """
    Class for Isomap dimensionality reduction.
    """

    def __init__(self, data, n_components=2, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self.n_components = n_components

    def reduce_dimensionality(self):
        """
        Reduce the dimensionality of the input data using Isomap.

        Returns:
            np.array: The reduced data as a NumPy array.
        """
        preprocessed_data = self.preprocess_data(self.data)
        isomap = Isomap(n_components=self.n_components)
        reduced_data = isomap.fit_transform(preprocessed_data)
        return reduced_data

    def fit_transform(self):
        """
        Apply the Isomap dimensionality reduction method and return the reduced data.

        Returns:
            np.array: The reduced data as a NumPy array.
        """
        isomap = Isomap(n_components=2)
        reduced_data = isomap.fit_transform(self.data)
        return reduced_data
