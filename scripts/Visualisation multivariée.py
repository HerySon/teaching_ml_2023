#######################################
#Visualisation multivariée des données#
#######################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def create_scatter_matrix(df, hue=None, title=None):
    """
    Create a scatter plot matrix of a pandas dataframe.
    
    Parameters:
        df (pandas dataframe): The dataframe to create the scatter plot matrix from.
        hue (string): The name of the column to use for the color of the markers. Default is None.
        title (string): The title of the plot. Default is None.
    
    Returns:
        None
    """
    sns.pairplot(data=df, hue=hue)
    plt.title(title)
    plt.show()


def create_heatmap(df, annot=True, fmt=".2f", cmap="coolwarm", title=None):
    """
    Create a heatmap of the correlation matrix of a pandas dataframe.
    
    Parameters:
        df (pandas dataframe): The dataframe to create the heatmap from.
        annot (bool): Whether or not to annotate the heatmap. Default is True.
        fmt (string): The format string to use for the annotations. Default is ".2f".
        cmap (string): The name of the colormap to use. Default is "coolwarm".
        title (string): The title of the plot. Default is None.
    
    Returns:
        None
    """
    corr = df.corr()
    sns.heatmap(corr, annot=annot, fmt=fmt, cmap=cmap)
    plt.title(title)
    plt.show()


def create_dendrogram(df, linkage_method='ward', metric='euclidean', color_threshold=0, title=None):
    """
    Create a dendrogram of a pandas dataframe.
    
    Parameters:
        df (pandas dataframe): The dataframe to create the dendrogram from.
        linkage_method (string): The linkage method to use. Must be one of 'ward', 'complete', 'average', 'single', or 'weighted'.
            Default is 'ward'.
        metric (string): The distance metric to use. Default is 'euclidean'.
        color_threshold (float): The threshold to use for coloring the dendrogram. Default is 0.
        title (string): The title of the plot. Default is None.
    
    Returns:
        None
    """
    from scipy.cluster.hierarchy import dendrogram, linkage
    linkage_matrix = linkage(df, method=linkage_method, metric=metric)
    dendrogram(linkage_matrix, color_threshold=color_threshold)
    plt.title(title)
    plt.show()
