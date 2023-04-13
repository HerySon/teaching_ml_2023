import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
from category_encoders import CountEncoder, QuantileEncoder



class DataEncoder:
    def __init__(self) -> None:
        pass

    def _check_categorical_data(self, datas: pd.DataFrame) -> pd.DataFrame:
        """
        Helper method to identify and return only the categorical columns from the input DataFrame.

        Args:
            datas (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing only the categorical columns.
        """
        categorical_columns = datas.select_dtypes(include=['object', 'category']).columns
        return datas[categorical_columns]

    def one_hot_encoding(self, datas: pd.DataFrame, sparse: bool = False) -> pd.DataFrame:
        """
        Performs one-hot encoding on the categorical columns of the input DataFrame.

        Args:
            datas (pd.DataFrame): Input DataFrame.
            sparse (bool, optional): Whether to return a sparse matrix. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame with one-hot encoded columns.
        """
        datas = self._check_categorical_data(datas)
        encoder = OneHotEncoder(sparse=sparse)
        encoded_datas = encoder.fit_transform(datas)
        encoded_columns = encoder.get_feature_names_out(datas.columns)
        encoded_df = pd.DataFrame(encoded_datas, columns=encoded_columns)
        return encoded_df

    def feature_hashing(self, datas: pd.DataFrame, n_features: int = 10) -> pd.DataFrame:
        """
        Performs feature hashing on the categorical columns of the input DataFrame.

        Args:
            datas (pd.DataFrame): Input DataFrame.
            n_features (int, optional): Number of features to hash. Defaults to 10.

        Returns:
            pd.DataFrame: DataFrame with hashed columns.
        """
        datas = self._check_categorical_data(datas)
        hasher = FeatureHasher(n_features=n_features, input_type='string')
        hashed_features = hasher.fit_transform(datas.astype(str).values)
        hashed_df = pd.DataFrame(hashed_features.toarray())
        return hashed_df

    def count_encoding(self, datas: pd.DataFrame) -> pd.DataFrame:
        """
        Performs count encoding on the categorical columns of the input DataFrame.

        Args:
            datas (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with count encoded columns.
        """
        datas = self._check_categorical_data(datas)
        encoder = CountEncoder()
        encoded_datas = encoder.fit_transform(datas)
        return encoded_datas

    def quantile_encoding(self, datas: pd.DataFrame, n_quantiles: int = 10) -> pd.DataFrame:
        """
        Performs quantile encoding on the categorical columns of the input DataFrame.

        Args:
            datas (pd.DataFrame): Input DataFrame.
            n_quantiles (int, optional): Number of quantiles to use for encoding. Defaults to 10.

        Returns:
            pd.DataFrame: DataFrame with quantile encoded columns.
        """
        datas = self._check_categorical_data(datas)
        encoder = QuantileEncoder(n_quantiles=n_quantiles)
        encoded_datas = encoder.fit_transform(datas)
        return encoded_datas
