
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
