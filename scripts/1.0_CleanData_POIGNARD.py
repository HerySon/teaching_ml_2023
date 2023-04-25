## Used libraries

import pandas as pd

#####
## Functions use to clean the Dataset
#####

def drop_columns(df, cols_to_drop):
    """
    Drop specific Column wich are not usefull for our Project

    Parameter :
    df(DataFrame) : The dataframe
    cols_to_drop(list) : list of specified columns to drop

    Returns : 
    Dataframe : cleaned dataset
    """
    df = df.drop(cols_to_drop, axis=1)
    return df

def drop_missing_data(df, key_cols):
    """
    Delete all the rows with missing data in specified key columns. 

    Paramaters : 
    df(DataFrame): The dataset
    key_cols(list) : list of specified columns

    Returns : 
    DataFrame : cleaned dataset
    """
    df = df.dropna(subset=key_cols)
    return df

def drop_duplicates(df, key_cols):
    """
    Drop all duplicates in specified columns keys 

    Paramters : 
    df(DataFrame) : The dataset 
    key_cols(list): list of specified columns

    DataFrame : cleaned dataset
    """
    df = df.drop_duplicates(subset=key_cols)
    return df

def convert_to_numeric(df, num_cols):
    """
    Convert the specified columns to int or float
    Parameters : 
    df(DataFrame): The Dataset
    num_cols(list): list of specified columns

    Returns : 
    DataFrame : cleaned dataset
    """
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    return df

def replace_outliers(df, outliers_dict):
    """
    Replace outliers by missing data (NaN) for specified columns

    Parameters : 
    df(DataFrame): The Dataset
    outliers_dict(list) : List of outliers we want to replace
    """
    for col, value in outliers_dict.items():
        df.loc[df[col] > value, col] = pd.NA
    return df

def drop_missing_columns(df, threshold):
    """
    Delete columns with too much missing data with a specified threshold

    Parameters : 
    df(DataFrame): The Dataset
    threshold(float): the threshold to use 
    """
    df = df.dropna(thresh=threshold*len(df), axis=1)
    return df
