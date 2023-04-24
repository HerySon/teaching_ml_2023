import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def reduce_dimensions(data, method, features=None, perplexity=None, random_state=None):
    """
        -----------------------------------------------------------------------
        Reduces the number of dimensions in the input 
        data using the specified method.
        -----------------------------------------------------------------------
        Parameters:
        -----------------------------------------------------------------------
        - data (pandas.DataFrame): The input data to reduce dimensions on.
        - method (str): The dimensionality reduction method to use. 
        Either "pca" or "tsne".
        - features (list of str, optional): The feature columns to use for 
        dimensionality reduction. Required if method is "pca".
        - perplexity (float, optional): The perplexity hyperparameter for t-SNE.
        Required if method is "tsne".
        - random_state (int, optional): The random state for reproducibility.
        Required if method is "tsne".
        ---------------------------------------------------------------------
        Returns:
        ---------------------------------------------------------------------
        The reduced dataset.
        ---------------------------------------------------------------------
    """
    if method == 'pca':
        # Perform PCA to reduce dimensions
        pca = PCA(n_components=2) # reduce to 2 dimensions for visualization
        reduced_data = pd.DataFrame(pca.fit_transform(data[features]), columns=['PC1', 'PC2'])
    elif method == 'tsne':
        # Perform t-SNE to reduce dimensions
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
        reduced_data = pd.DataFrame(tsne.fit_transform(data[features]), columns=['t-SNE 1', 't-SNE 2'])
    else:
        raise ValueError('Unsupported dimensionality reduction method: {}'.format(method))
    
    return reduced_data
    


if __name__ == '__main__':

    # Select some columns to use as features
    features = ['energy_100g', 'fat_100g', 'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g']

    # Reduce dimensions using PCA
    reduced_data_pca = reduce_dimensions(data, 'pca', features=features)

    # Reduce dimensions using t-SNE
    reduced_data_tsne = reduce_dimensions(data, 'tsne', features=features, perplexity=30, random_state=42)
    # print the shape of the reduced datasets
    print('PCA reduced dataset shape:', reduced_data_pca.shape)
    print('t-SNE reduced dataset shape:', reduced_data_tsne.shape)