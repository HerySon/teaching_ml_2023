#import pandas as pd
#import matplotlib.pyplot as plt
#from wordcloud import WordCloud
#import unittest


def plot_wordcloud(data, column):
    """
    Displays a wordcloud for the given column.

    Args:
        data (DataFrame): The dataframe to use.
        column (str): The name of the column to display.
    """
    text = ' '.join(data[column].astype(str).tolist())
    wordcloud = WordCloud(width=800, height=800, background_color='white', colormap='Set2', min_font_size=10).generate(text)

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()


def plot_density(data, column):
    """
    Displays a density plot for the given column.

    Args:
        data (DataFrame): The dataframe to use.
        column (str): The name of the column to display.
    """
    data[column].plot(kind='density')
    plt.show()


def plot_unique_values(data):
    """
    Displays a barplot for each column of the given dataframe,
    showing the number of unique values for each column.

    Args:
        data (DataFrame): The dataframe to use.
    """
    unique_vals = []
    for col in data.columns:
        unique_vals.append(data[col].nunique())

    plt.bar(data.columns, unique_vals)
    plt.xticks(rotation=90)
    plt.xlabel('Columns')
    plt.ylabel('Number of unique values')
    plt.show()


def plot_histogram(data, column):
    """
    Displays a histogram for the given column.

    Args:
        data (DataFrame): The dataframe to use.
        column (str): The name of the column to display.
    """
    data[column].hist()
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

    """
    unit test for each fonctions bellow
    """

    class TestVisualizationFunctions(unittest.TestCase):

      def setUp(self):
        # Create a sample dataframe for testing
        self.df = pd.DataFrame({
            'col1': ['florent', 'joelle', 'nicolas', 'thomas'],
            'col2': [1, 2, 3, 4],
            'col3': ['a', 'b', 'c', 'd'],
            'col4': [1.0, 2.0, 3.0, 4.0]
        })

    def test_plot_wordcloud(self):
        # Test if the function wordcloud runs without errors
        plot_wordcloud(self.df, 'col1')

    def test_plot_density(self):
        # Test if the function ednsity runs without errors
        plot_density(self.df, 'col2')

    def test_plot_unique_values(self):
        # Test if the function values runs without errors
        plot_unique_values(self.df)

    def test_plot_histogram(self):
        # Test if the function histogram runs without errors
        plot_histogram(self.df, 'col4')
