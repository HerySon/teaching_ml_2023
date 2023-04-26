import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def tSNE_reduce_methode(X, n_iter = 1000, n_components = 2,
                        perplexity = 30, init = "random", 
                        learning_rate='auto',**kwags):
    
    """The purpose of t-SNE is visualization of high-dimensional data. 
    It works best when the data will be embedded on two or three dimensions.
    Args:
        X: {array-like, sparse matrix} of shape (n_samples, n_features) or (n_samples, n_samples)
        n_iterint(int) = Maximum number of iterations for the optimization. At least 250. 
        n_components(int): number of components after reduction
        perplexity(float): Nearest neighbors betwen 5 to 50
        intit({“random”, “pca”} or ndarray of shape (n_samples, n_components)) = Initialization of embedding
        learning_rate (float or "auto"): step for gradient descent;
                                        range [10.0, 1000.0].
    Returns:
        array: Embedding of the training data in low-dimensional space
    Made By: Florent Sanchez
    """
    n_iter = kwargs.get("n_iter", 1000) 
    n_components = kwargs.get("n_components", 2)
    perplexity = kwargs.get("perplexity", 30)
    init = kwargs.get("init", 'random')
    learning_rate = kwargs.get("learning_rate", "auto")  
    
    tsne = TSNE(n_iter = n_iter,
                n_components = n_components, 
                perplexity = perplexity,
                random_state = 42, 
                init = init,
                learning_rate = learning_rate)
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