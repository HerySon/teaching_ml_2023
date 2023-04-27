#Dataframe
import pandas as pd
import numpy as np
#Viz
import matplotlib.pyplot as plt
#Chi2 test
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest


def cross_table (df,index,columns) : 
    """
    Explore variance of the categorial features in a pandas dataframe using pandas.
    Args:
        df : pandas dataframe
        index : str (column name to plot)
        columns : str (column to plot)

    Returns : 
        Cross table
    """
    #Cross tables
    data_crosstab = pd.crosstab(df.index, df.columns,
                            margins = False)
    return print(data_crosstab)


def categorial_feature_selection(k,X, y):
    """
    This score can be used to select the n_features features with the highest values.
    must contain only non-negative features.
    
    Arg:
    k(int): top features to be selected
    X: {array-like, sparse matrix} of shape (n_samples, n_features)
    y: array-like of shape (n_samples)
    
    Return
    

    """

    # k tells top features to be selected
    # Score function Chi2 tells the feature to be selected using Chi Square
    feature_score = SelectKBest(score_func=chi2, k=k)
    fit = model.fit(X, y)
    scores = fit.scores_
    X_select = model.fit_transform(X, y)
    return scores_,X_select

def plot_univariate_feature_selection(X,scores,width):
    """
    Select the most significant features.
    
    Arg: X: {array-like, sparse matrix}
        scores: array-like of shape (n_features,)
        width (float)
    
    Return: plot 
    """
    X_indices = np.arange(X.shape[-1])
    score = fit.scores_
    plt.figure(1)
    plt.clf()
    plt.bar(X_indices, scores, width=width)
    plt.title("Feature univariate score")
    plt.xlabel("Feature number")
    plt.ylabel("Univariate score")
    return plt.show()