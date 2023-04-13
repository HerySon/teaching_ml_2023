from sklearn.manifold import MDS
from DimensionalityReduction import DimensionalityReduction


class MDSReduction(DimensionalityReduction):
    """
    Class for Multi-Dimensional Scaling (MDS) dimensionality reduction.
    """

    def __init__(self, data, n_components=2, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self.n_components = n_components

    def reduce_dimensionality(self):
        """
        Reduce the dimensionality of the input data using MDS.

        Returns:
            np.array: The reduced data as a NumPy array.
        """
        preprocessed_data = self.preprocess_data(self.data)
        mds = MDS(n_components=self.n_components)
        reduced_data = mds.fit_transform(preprocessed_data)
        return reduced_data

    def fit_transform(self):
        """
        Apply the MDS dimensionality reduction method and return the reduced data.

        Returns:
            np.array: The reduced data as a NumPy array.
        """
        mds = MDS(n_components=2)
        reduced_data = mds.fit_transform(self.data)
        return reduced_data
