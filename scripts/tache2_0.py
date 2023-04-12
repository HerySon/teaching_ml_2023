import data_loader as dl
import pandas as pd
import numpy as np


from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

class DataScaler:
    def __init__(self) -> None:
        pass
    
    def _check_numerical_data(self, datas: pd.DataFrame):
        """
        Check that all columns in the DataFrame are numerical. If not, raises a ValueError exception.

        :param datas: pd.DataFrame
            DataFrame whose columns should be checked.
        """
        if not all(datas.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
            raise ValueError("the whole dataframe needs to be numerical, please do tache 1.1 before ")
        
    def min_max_scaling(self, datas: pd.DataFrame, feature_range=(0, 1)) -> pd.DataFrame:
        """
        Applies Min-Max scaling to the input DataFrame.

        :param datas: pd.DataFrame
            DataFrame to be scaled.
        :param feature_range: tuple, optional
            Desired range of transformed data.
        :return: pd.DataFrame
            Scaled DataFrame.
        """
        scaler = MinMaxScaler(feature_range=feature_range)
        scaled_datas = pd.DataFrame(scaler.fit_transform(datas), columns=datas.columns)
        return scaled_datas

    def standard_scaling(self, datas: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Standard scaling (also known as Z-score scaling) to the input DataFrame.

        :param datas: pd.DataFrame
            DataFrame to be scaled.
        :return: pd.DataFrame
            Scaled DataFrame.
        """
        scaler = StandardScaler()
        scaled_datas = pd.DataFrame(scaler.fit_transform(datas), columns=datas.columns)
        return scaled_datas

    def robust_scaling(self, datas: pd.DataFrame, quantile_range=(25, 75)) -> pd.DataFrame:
        """
        Applies Robust scaling to the input DataFrame, which is less influenced by outliers.

        :param datas: pd.DataFrame
            DataFrame to be scaled.
        :param quantile_range: tuple, optional
            Quantile range used for scaling.
        :return: pd.DataFrame
            Scaled DataFrame.
        """
        scaler = RobustScaler(quantile_range=quantile_range)
        scaled_datas = pd.DataFrame(scaler.fit_transform(datas), columns=datas.columns)
        return scaled_datas

    def max_abs_scaling(self, datas: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Max-Absolute scaling to the input DataFrame, which scales the data by its maximum absolute value.

        :param datas: pd.DataFrame
            DataFrame to be scaled.
        :return: pd.DataFrame
            Scaled DataFrame.
        """
        scaler = MaxAbsScaler()
        scaled_datas = pd.DataFrame(scaler.fit_transform(datas), columns=datas.columns)
        return scaled_datas







   