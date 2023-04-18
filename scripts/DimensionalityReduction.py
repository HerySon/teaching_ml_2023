import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler


class DimensionalityReduction:
    """
    Main class for dimensionality reduction methods.
    """

    def __init__(self, data, n_clusters=10):
        """
        Initialize the DimensionalityReduction class with data and optional parameters.

        Args:
            data (pd.DataFrame): The input data as a Pandas DataFrame.
            n_clusters (int, optional): The number of clusters for coloring the plot. Defaults to 10.
        """
        self.data = data
        self.n_clusters = n_clusters

    @staticmethod
    def preprocess_data(data):
        """
        Preprocess the input data by converting non-numeric columns to numeric using label encoding,
        dropping rows with NaN values, and scaling the data.

        Returns:
            pd.DataFrame: The preprocessed data as a Pandas DataFrame.
        """

        # Convert non-numeric columns to numeric using label encoding
        for column in data.columns:
            if data[column].dtype not in ['int64', 'float64']:
                le = LabelEncoder()
                data[column] = le.fit_transform(data[column].astype(str))

        # Fill missing values with the mean for numeric columns and mode for non-numeric columns
        for column in data.columns:
            if data[column].dtype in ['int64', 'float64']:
                data[column] = data[column].fillna(data[column].mean())
            else:
                data[column] = data[column].fillna(data[column].mode().iloc[0])

        data = data.dropna(axis=1, how='all')

        print(data)

        # Scale the data using StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        # Normalize the data using MinMaxScaler
        normalizer = MinMaxScaler()
        normalized_data = normalizer.fit_transform(scaled_data)

        # Convert back to DataFrame
        normalized_data = pd.DataFrame(normalized_data, columns=data.columns)

        return normalized_data

    @staticmethod
    def plot(reduced_data, method_name):
        """
        Plot the reduced data using a scatter plot.

        Args:
            reduced_data (pd.DataFrame): The reduced data as a Pandas DataFrame.
            method_name (str): The name of the dimensionality reduction method used for the plot title.
        """
        plt.figure(figsize=(10, 10))
        plt.title(f'{method_name} Visualization of the Data')
        plt.xlabel('First Component')
        plt.ylabel('Second Component')
        plt.show()
