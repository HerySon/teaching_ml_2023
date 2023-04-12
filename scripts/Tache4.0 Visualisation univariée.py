######################################
#Visualisation des données univariés #
######################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def create_wordcloud(df, col, stopwords=None):
    """
    Create a wordcloud of unique values in a given column of a pandas dataframe.
    
    Parameters:
        df (pandas dataframe): The dataframe to create the wordcloud from.
        col (string): The name of the column to create the wordcloud from.
        stopwords (list): A list of stopwords to exclude from the wordcloud. Default is None.
    
    Returns:
        None
    """
    if stopwords is None:
        stopwords = []
    text = " ".join(df[col].astype(str))
    wordcloud = WordCloud(stopwords=stopwords).generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def create_density_plot(df, col, xlim=None, title=None):
    """
    Create a density plot of a given column in a pandas dataframe.
    
    Parameters:
        df (pandas dataframe): The dataframe to create the density plot from.
        col (string): The name of the column to create the density plot from.
        xlim (tuple): A tuple containing the limits of the x-axis. Default is None.
        title (string): The title of the plot. Default is None.
    
    Returns:
        None
    """
    df[col].plot(kind="density")
    plt.xlim(xlim)
    plt.title(title)
    plt.show()


def create_value_counts(df, col, title=None, barplot=True):
    """
    Count the unique values in a given column of a pandas dataframe and create a barplot or histogram.
    
    Parameters:
        df (pandas dataframe): The dataframe to count the unique values from.
        col (string): The name of the column to count the unique values from.
        title (string): The title of the plot. Default is None.
        barplot (boolean): If True, create a barplot. If False, create a histogram. Default is True.
    
    Returns:
        None
    """
    counts = df[col].value_counts()
    if barplot:
        counts.plot(kind="bar")
    else:
        counts.plot(kind="hist")
    plt.title(title)
    plt.show()
