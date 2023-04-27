 
#Dataframe
import pandas as pd
import numpy as np
#Viz
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
#Prepocessing
from sklearn import preprocessing
#PCA
from sklearn.decomposition import PCA

def numerical_correlation (df,columns) : 
    """
    Explore variance of the numerical features in a pandas dataframe using matplotlib.
    The input must not contains Nan value.
    Args:
        df : pandas dataframe
        columns : list of str (The list of column names to plot)

    Returns : 
        Plot of the variance matrice
        PCA 
    """
    #Select numerical features
    num_val = df[columns].select_dtypes(include='number')
    # Triangle mask
    fig = num_val[columns].corr(method='spearman').style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'))
    return fig

def pca_numerical (df, n_components): 
    """
    Explore correlation of the numerical features in a pandas dataframe using sklearn for PCA.
    The input must not contains Nan value.
    Args:
        df : pandas dataframe
        n_components : components of the PCA

    Returns : 
        PCA 
    """
    #Select numerical value
    num_val = df[columns].select_dtypes(include=['number'])
    #Scale the features
    std_scale = preprocessing.StandardScaler().fit(num_val)
    X_scaled = std_scale.transform(num_val)
    #PCA
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(X_scaled)
    columns = pca.get_feature_names_out()
    principalDf = pd.DataFrame(data = principalComponents, columns = columns)
    return principalDf,pca



def plot_cumulative_variances (pca):
    """
    Plot the informations of the PCA 
    Args:
        cumulative_variances
        eigenvalues
    Returns : 
        Plot the PCA informations
    """
    # Calculate cumulate values of proportional variance
    cumulative_variances = np.cumsum(explained_variances)
     # Determine eignten_value and proportional variance
    eigenvalues = pca.explained_variance_
    explained_variances = pca.explained_variance_ratio_
    # Plot of cumulate eigenvalues
    plt.subplot(1,2,1)
    plt.plot(range(1, len(cumulative_variances) + 1), cumulative_variances, marker='o')
    plt.title("Cumulate eigtenvalues")
    plt.xlabel("Number Principale Composant")
    plt.ylabel("Proportion of cumulate variance")
    # Plot proportionalcumulate variance
    plt.subplot(1,2,2)
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o')
    plt.title("Proportional variance")
    plt.xlabel("Principale Composant")
    plt.ylabel("Eigenvalues")
    #plt.tight_layout()
    plt.subplots_adjust(wspace=2.5, hspace=4)
    return plt.show()