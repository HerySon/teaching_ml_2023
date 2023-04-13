import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from data_loader import *
### In this script we use as reusable functions, the different scalling methods

def min_max_scaling(df, columns):
    """
    Min-Max scaling on specified numeric columns of the DataFrame

    Parameters:
        df(DataFrame): The dataset to use
        columns(list): List of columns we want

    Returns: 
        DataFrame : the scaled dataset 
    """
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[columns])
    df_scaled = pd.DataFrame(df_scaled, columns=columns)
    return df_scaled


def standard_scaling(df, columns):
    """
    Standard scaling of the DataFrame.  

    Parameters:
        df(DataFrame): The dataset to use
        columns(list): List of columns we want

    Returns: 
        DataFrame : the scaled dataset 
    """
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[columns])
    df_scaled = pd.DataFrame(df_scaled, columns=columns)
    return df_scaled


def robust_scaling(df, columns):
    """
    Robust scaling of the DataFrame. 

    Parameters:
        df(DataFrame): The dataset to use
        columns(list): List of columns we want

    Returns: 
        DataFrame : the scaled dataset 
    """
    scaler = RobustScaler()
    df_scaled = scaler.fit_transform(df[columns])
    df_scaled = pd.DataFrame(df_scaled, columns=columns)
    return df_scaled

def scale_transform_df(df, scaled_df):
    """
    Transform dataset with provided scaler

    Parameters:
        df (DataFrame): dataframe
        scaled_df (Scaler): fitted scaler 

    Returns:
        dataframe: output scaled dataframe
    """
    df_cols = df.select_dtypes([np.number]).columns
    df[df_cols] = scaled_df.transform(df[df_cols])
    return df

