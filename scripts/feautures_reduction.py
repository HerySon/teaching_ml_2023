import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def tSNE_reduce_methode(X, n_components, perplexity):
    """The purpose of t-SNE is visualization of high-dimensional data. 
    It works best when the data will be embedded on two or three dimensions.
    Args:
        X: {array-like, sparse matrix} of shape (n_samples, n_features) or (n_samples, n_samples)
        n_components(int): number of components after reduction
        perplexity(float): Nearest neighbors betwen 5 to 50
    Returns:
        array: Embedding of the training data in low-dimensional space
    """
    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                random_state=42, init= 'random')
    X_tsne = tsne.fit_transform(X)
    return X_tsne

def plot_reduced_features(X_tsne):
    """Visualization of high-dimensional data. 
    Args:
        X_tsne: array of shape (n_samples, n_components)
    Returns:
        Plot: scatter
    """ 
       
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
    return plt.show()