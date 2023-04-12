import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from data_loader import get_data
from sklearn.cluster import KMeans
from utils import drop_cols_fullna, std_scale_df, get_num_feats

def fit_tsne(df, perplexity=50, init='pca', n_components=2, learning_rate='auto', n_iter=1000, n_jobs=-1, **kwargs):
    """Output tnse fitted model with provided hyperparameters and df
    Args:
        df (DataFrame): pandas dataframe
        perplexity (integer, optional): tsne hyperparameter
        init (string | ndarray, optional): tsne hyperparameter
        n_components (integer, optional): tsne hyperparameter
        learning_rate (float | string , optional): tsne hyperparameter
        n_iter (int , optional): tsne hyperparameter
        kwargs (): Hyperparameters of tsne
    Returns:
        tsne: Return fitted tsne model
    @Author: Nicolas THAIZE
    """
    early_exaggeration = kwargs.get("early_exaggeration", 12.0)
    n_iter_without_progress = kwargs.get("n_iter_without_progress", 300)
    min_grad_norm = kwargs.get("min_grad_norm", 1e-7)
    metric = kwargs.get("metric", "euclidean")
    metric_params = kwargs.get("metric_params", None)
    verbose = kwargs.get("verbose", 0)
    random_state = kwargs.get("random_state", None)
    method = kwargs.get("method", "barnes_hut")
    angle = kwargs.get("angle", 0.5)
    
    tsne = TSNE(
        n_components=n_components, 
        perplexity=perplexity, 
        init=init, 
        learning_rate=learning_rate, 
        n_iter=n_iter, 
        n_jobs=n_jobs,
        early_exaggeration=early_exaggeration,
        n_iter_without_progress=n_iter_without_progress,
        min_grad_norm=min_grad_norm,
        metric=metric,
        metric_params=metric_params,
        verbose=verbose,
        random_state=random_state,
        method=method,
        angle=angle
        ).fit(df)
    return tsne

def transform_tnse(tsne, df):
    """Transform dataset using fitted tsne model
    Args:
        tsne (tsne): Fitted tsne model
        df (DataFrame): pandas dataframe
    Returns:
        tsne: Return transformed df
    @Author: Nicolas THAIZE
    """
    return tsne.fit_transform(df)

def plot_tsne(tsne_matrix, clusters_array, cmap=plt.get_cmap('tab20c')):
    """Plot tsne transformed df
    Args:
        tsne_matrix (ndarray): 2D matrix of transformed values from transform_tsne
        clusters_array (ndarray): 1D matrix of cluster labels for each entries
        cmap (Colormap): colors to use to paint clusters
    Returns:
    @Author: Nicolas THAIZE
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    for i in range(tsne_matrix.shape[0]):
        cval = cmap(clusters_array[i])
        ax.scatter(tsne_matrix[i][0], tsne_matrix[i][1], marker='.', color=cval)
        ax_title = 'Représentation en 2D du dataset avec clustering'
        ax.set_title(ax_title)
        ax.set_xlabel('premier t-SNE')
        ax.set_ylabel('deuxieme t-SNE')

    plt.plot()
    plt.show()

def fit_multiple_perplexity(perplexities, df, init='pca', n_components=2, learning_rate='auto', n_iter=1000):
    """Output an array of tnse fitted models with provided perplexities values
    Args:
        perplexities (array integer): array of perplexity values to use to fit tsne models
        df (DataFrame): pandas dataframe
        init (string | ndarray, optional): tsne hyperparameter
        n_components (integer, optional): tsne hyperparameter
        learning_rate (float | string , optional): tsne hyperparameter
        n_iter (int , optional): tsne hyperparameter
        kwargs (): Hyperparameters of tsne
    Returns:
        tsne: Return fitted tsne model
    @Author: Nicolas THAIZE
    """
    tsne_perplexity = perplexities
    results = {}
    for perplexity in tsne_perplexity:
        results[perplexity] = fit_tsne(df, n_components=n_components, perplexity=perplexity, init=init, learning_rate=learning_rate, n_iter=n_iter)
    return results

def transform_multiple_tsne(tsnes, df):
    """Transform dataset using multiple fitted tsne models
    Args:
        tsnes (array tsne): Fitted tsne models
        df (DataFrame): pandas dataframe
    Returns:
        results (dict): Return transformed df for each tsne provided
    @Author: Nicolas THAIZE
    """
    results = {}
    for key, tsne in tsnes.items():
        results[key] = transform_tnse(tsne, df)
    return results

def plot_tsne_perplexity(perplexities, df, clusters_array, cmap=plt.get_cmap('tab20c'), **kwargs):
    """Plot tsne using multiple perplexities
    Args:
        perplexities (array): perplexities to use to plot
        df (DataFrame): pandas dataframe
        clusters_array (ndarray): 1D matrix of cluster labels for each entries
        cmap (Colormap): colors to use to paint clusters
        kwargs (): Hyperparameters of tsne
    Returns:
    @Author: Nicolas THAIZE
    """
    perplexity = kwargs.get("perplexity", 50)
    init = kwargs.get("init", 'pca')
    n_components = kwargs.get("n_components", 2)
    learning_rate = kwargs.get("learning_rate", "auto")  
    n_iter = kwargs.get("n_iter", 1000)  
    
    tsne_perplexity = perplexities
    results = {}
    
    for perplexity in tsne_perplexity:
        tsnes = fit_multiple_perplexity(tsne_perplexity, df, n_components=n_components, init=init, learning_rate=learning_rate, n_iter=n_iter)
        results = transform_multiple_tsne(tsnes, df)
        fig, axs = plt.subplots(len(perplexities), figsize=(30, 30))

        for index, perplexity in enumerate(results):
            currentAxis = axs[index]
            X_tsne = results[perplexity]
            for i in range(X_tsne.shape[0]):
                cval = cmap(clusters_array[i])
                currentAxis.scatter(X_tsne[i][0], X_tsne[i][1], marker='.', color=cval)
                ax_title = 'Représentation en 2D du dataset avec cluster perplexity = ' + str(perplexity)
                currentAxis.set_title(ax_title)
                currentAxis.set_xlabel('premier t-SNE')
                currentAxis.set_ylabel('deuxieme t-SNE')
    plt.plot()
    plt.show()

def get_kl_divergence_score(tsne):
    """Return Kullback-Leibler divergence score 
    Args:
        tsne (tsne): Fitted tsne model
    Returns:
        kl_divergence_: Return kl divergence score for provided tsne fitted&transformed model
    @Author: Nicolas THAIZE
    """
    return tsne.kl_divergence_



if __name__ == "__main__":
    data = get_data(file_path = "../data/en.openfoodfacts.org.products.csv", nrows=1000)
    
    #Data preparation
    tsne_df = get_num_feats(data)
    tsne_df = drop_cols_fullna(tsne_df)
    tsne_df = tsne_df.fillna(tsne_df.mean())
    tsne_df = std_scale_df(tsne_df)

    # Naive clustering, should be remplaced by elaborated clustering 
    kmeans = KMeans(n_clusters=10, random_state=0)
    kmeans_result = kmeans.fit_predict(tsne_df)


    ## Basic tsne
    tsne = fit_tsne(tsne_df)
    result = transform_tnse(tsne=tsne, df=tsne_df)
    plot_tsne(result, kmeans_result)
    print("Kullback-Leibler divergence score : " + str(get_kl_divergence_score(tsne)))

    # Plotting multiple tsne based on perplexity hyperparameter
    plot_tsne_perplexity([10, 50, 100], tsne_df, kmeans_result)
