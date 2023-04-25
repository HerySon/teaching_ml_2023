import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def wordcloud_feature(df: pd.DataFrame, feature: str, stopwords: list = None):
    """
    Generates a WordCloud from the values ​​of a given feature.

    args:
        df (pd.DataFrame): The dataframe containing the feature.
        feature (str): The name of the feature to use.
        stopwords (list): The list of words to ignore in WordCloud generation.

    Returns:
        None
    """
    # Verification of the existence of stopwords
    if stopwords is None:
        stopwords = []

    # Creation of the list of words from the data of the feature
    words = ' '.join(df[feature].dropna().astype(str).values.tolist())

    # WordCloud generation
    wordcloud = WordCloud(width=800, height=400, stopwords=stopwords).generate(words)

    # WordCloud display
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Feature WordCloud '{feature}'", fontsize=20, fontweight='bold')
    plt.show()

def density(data: pd.Series, title: str = None, xlabel: str = None, ylabel: str = 'Density') -> None:
    """
    Displays the density graph for a numeric variable.

    args:
        data (pd.Series): The numeric variable to analyze.
        title (str): The chart title. Default is None.
        xlabel (str): The x axis label. Default is None.
        ylabel (str): The y-axis label. Default is 'Density'.
    
    Returns:
        None
    """
    plt.figure(figsize=(10,6))
    sns.kdeplot(data)
    plt.title(title or f"Density Plot of {data.name}", fontsize=16)
    plt.xlabel(xlabel or data.name, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.show()

def unique_values(data: pd.DataFrame, column: str, max_values: int = None) -> None:
    """
    Displays the unique values ​​of a categorical variable.

    args:
        data (pd.DataFrame): The dataframe containing the variable to analyze.
        column (str): The name of the column to analyze.
        max_values ​​(int): The maximum number of values ​​to display. Default is None.
    
    Returns:
        None
    """
    values = data[column].unique()
    if max_values:
        values = values[:max_values]
    print(f"Unique values for {column}: {values}")


def plot_feature(df: pd.DataFrame, feature_name: str, plot_type: str, **kwargs) -> None:
    """
    Creates a histogram or barplot chart for the given feature.

    args:
    - df: pandas dataframe containing the column to represent
    - feature_name: name of the column to represent
    - plot_type: type of graph to create (histogram or barplot)
    - **kwargs: additional arguments for the plot function (for example, the number of bins for a histogram)

    Returns:
    - None
    """

    if plot_type == 'hist':
        plt.hist(df[feature_name], **kwargs)
    elif plot_type == 'bar':
        sns.countplot(df[feature_name], **kwargs)
    else:
        raise ValueError('Invalid plot_type argument. Allowed values are "hist" and "bar".')

    plt.title(feature_name)
    plt.xlabel(feature_name)
    plt.ylabel('Frequency')
    plt.show()
