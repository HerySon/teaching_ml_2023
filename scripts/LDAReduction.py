from sklearn.decomposition import LatentDirichletAllocation as LDA
from DimensionalityReduction import DimensionalityReduction


class LDAReduction(DimensionalityReduction):
    """
    Class for Latent Dirichlet Allocation (LDA) dimensionality reduction.
    """

    def __init__(self, data, n_components=2, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self.n_components = n_components

    def reduce_dimensionality(self):
        """
        Reduce the dimensionality of the input data using LDA.

        Returns:
            np.array: The reduced data as a NumPy array.
        """
        preprocessed_data = self.preprocess_data(self.data)
        lda = LDA(n_components=self.n_components)
        reduced_data = lda.fit_transform(preprocessed_data)
        return reduced_data

    def fit_transform(self):
        """
        Apply the LDA dimensionality reduction method and return the reduced data.

        Returns:
            np.array: The reduced data as a NumPy array.
        """
        lda = LDA(n_components=2)
        reduced_data = lda.fit_transform(self.data)
        return reduced_data
