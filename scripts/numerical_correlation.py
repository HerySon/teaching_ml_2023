 
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

def numerical (df,columns) : 
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
    num_val = df[columns].select_dtypes(include=['float64','int64','float32','int32'])
    # Triangle mask
    mask = np.triu(np.ones_like(num_val)) 
    #plot dimension
    fig = plt.figure(figsize=(20, 30))
    #heatmap configuration
    ax = sns.heatmap(num_val, mask=mask, vmin=-1, vmax=1,
                 square=True,linewidths=0.9,
                 annot=True, annot_kws={'fontsize':'large'}, cmap='BrBG')
    #Title
    ax.set_title('Correlation Heatmap', fontdict={'fontsize':20}, pad=5);
    #x_label
    ax.set_xlabel('', fontsize='10');
    plt.xticks(rotation = 45);
    return fig

def pca_numerical (df): 
    """
    Explore correlation of the numerical features in a pandas dataframe using matplotlib,seaborn and sklearn for PCA.
    The input must not contains Nan value.
    Args:
        df : pandas dataframe
        columns : numericale features

    Returns : 
        Plot of the variance matrice
        PCA 
    """
    #Select numerical value
    num_val = df[columns].select_dtypes(include=['float64','int64','float32','int32'])
    #Scale the features
    std_scale = preprocessing.StandardScaler().fit(num_val)
    X_scaled = std_scale.transform(num_val)
    #PCA
    pca = PCA(n_components=10)
    principalComponents = pca.fit_transform(X_scaled)
    principalDf = pd.DataFrame(df = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

def eigten_value (pca):
     """
    Determine eignten value. 
    Args:
        pca
    Returns : 
        Eigenvalues
        Explained Variance
    """
    # Determine eignten_value and proportional variance
    eigenvalues = pca.explained_variance_
    explained_variances = pca.explained_variance_ratio_
    return print('Eigenvalues are :'eigenvalues , 'Explained variances:' explained_variances)

def cumulative_variances(explained_variances):
    # Calculate cumulate values of proportional variance
    cumulative_variances = np.cumsum(explained_variances)
    return print('Cumulative Variances:' cumulative_variances)


def plot_cumulative_variances (cumulative_variances,eigenvalues):
    """
    Plot the informations of the PCA 
    Args:
        cumulative_variances
        eigenvalues
    Returns : 
        Plot the PCA informations
    """
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