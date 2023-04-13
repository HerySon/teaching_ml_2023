
#Required libraries
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud


#Plot a wordcloud to visualize the words of a given column with personalisable dimension

def plot_wordcloud(data, column,v_width,v_height):
    """
    Displays a wordcloud for the given column of pandas DataFrame .
    Args:
        data (DataFrame): The dataframe to use : At least one String type column
        column (str): The name of the column to display : String type
        v_width(int) : The width of the cloud.
        v_height(int) : The height of the cloud
    """
    if isinstance(data[column].iloc[0], str):
      text = " ".join(data[column].astype(str).tolist())
      wordcloud = WordCloud(v_width,v_height, background_color='white').generate(text)

      plt.figure(figsize=(7, 7), facecolor=None)
      plt.imshow(wordcloud)
      plt.show()
    else:
      raise ValueError("Column must be String type.")


def plot_density(data, column):
    """
    Displays a density plot.
    Args:
        data (DataFrame): The dataframe to use : At least one numeric column.
        column (int): The column to display : Numeric type.
    """
    if pd.api.types.is_numeric_dtype(data[column]):
      data[column].plot(kind='density')
      plt.show()
    else :
      raise ValueError("Column must be numeric type")


def plot_unique_values(data):
    """
    Displays a barplot for each columns represent
    the number of unique values.
    Args:
        data (DataFrame): The DataFrame : At least one numeric column.
    """
    cols = data.select_dtypes(include=['number']).columns
    if len(cols) == 0:
        raise ValueError("At least one column should be of numeric type")
    else:
        for col in cols:
            unique_values = data[col].nunique()
            plt.bar(col, unique_values)
        plt.xticks(rotation=90)
        plt.xlabel('Columns')
        plt.ylabel('Number of unique values')
        plt.show()

