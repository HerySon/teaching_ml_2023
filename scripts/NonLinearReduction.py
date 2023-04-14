from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.impute import SimpleImputer


class NonLinearReduction:
    """
    Class for non-linear dimensionality reduction using t-SNE.

    Parameters:
        data (pd.DataFrame): The input data as a Pandas DataFrame.
        n_components (int): The number of components to reduce the data to.
        perplexity (float): The perplexity parameter for t-SNE.
        random_state (int): The random seed for t-SNE.
        imputer_strategy (str): The imputation strategy for handling missing values.

    Returns:
        pd.DataFrame: The reduced data as a Pandas DataFrame.
    """

    def __init__(self, data, n_components=2, perplexity=5, random_state=42, imputer_strategy='mean'):
        self.data = data
        self.n_components = n_components
        self.perplexity = perplexity
        self.random_state = random_state
        self.imputer_strategy = imputer_strategy

    def reduce_dimensions(self):
        """
        Reduce the dimensionality of the input data using t-SNE.

        Returns:
            pd.DataFrame: The reduced data as a Pandas DataFrame.
        """
        # Preprocess data by scaling to zero mean and unit variance
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)

        # Impute missing values
        imputer = SimpleImputer(strategy=self.imputer_strategy)
        imputed_data = imputer.fit_transform(scaled_data)

        # Reduce dimensionality using t-SNE
        tsne = TSNE(n_components=self.n_components, perplexity=self.perplexity, random_state=self.random_state)
        reduced_data = tsne.fit_transform(imputed_data)

        # Convert back to Pandas DataFrame
        reduced_data = pd.DataFrame(reduced_data, columns=[f'component_{i + 1}' for i in range(self.n_components)])

        return reduced_data
