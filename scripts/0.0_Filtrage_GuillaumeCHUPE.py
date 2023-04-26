import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def filter_dataframe(df, num_cols=None, ordinal_cols=None, nominal_cols=None, downcast_cols=None, max_categories=None):
    """
    Filter and select relevant columns in the input data frame, distinguishing between numeric, ordinal, and nominal columns.

    Args:
        df (DataFrame): The input dataframe to filter.
        num_cols (list of str): List of column names for numeric columns. Default is None.
        ordinal_cols (list of str): List of column names for ordinal columns. Default is None.
        nominal_cols (list of str): List of column names for nominal columns. Default is None.
        downcast_cols (list of str): List of column names for columns that need to be downcasted. Default is None.
        max_categories (int): Maximum number of categories allowed for nominal columns. If None, all nominal columns are kept. Default is None.

    Returns:
        DataFrame: The filtered dataframe.
    """
    # Filter numeric columns
    if num_cols:
        numeric_cols = df.select_dtypes(include=['number']).columns.intersection(num_cols)
    else:
        numeric_cols = df.select_dtypes(include=['number']).columns

    # Filter ordinal columns
    if ordinal_cols:
        ordinal_cols = df[ordinal_cols].columns.intersection(ordinal_cols)
    else:
        ordinal_cols = []

    # Filter nominal columns
    if nominal_cols:
        if max_categories is not None:
            nominal_cols = df[nominal_cols].columns[df[nominal_cols].nunique() <= max_categories]
        else:
            nominal_cols = df[nominal_cols].columns
    else:
        nominal_cols = []

    # Downcast columns
    if downcast_cols:
        for col in downcast_cols:
            df[col] = pd.to_numeric(df[col], downcast='integer')

    # Return filtered dataframe with selected columns
    selected_cols = list(numeric_cols) + list(ordinal_cols) + list(nominal_cols)
    return df[selected_cols]
