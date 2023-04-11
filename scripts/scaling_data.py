import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def scaling_data(df: pd.DataFrame, how: str = 'standard', numeric_cols: list = None) -> pd.DataFrame:
    """
    Applies scaling to numeric columns of a DataFrame according to the chosen method.

    Parameters:
    -----------
    df: pd.DataFrame
        The DataFrame to scale.
    how: str, optional (default='standard')
        The scaling method to use. Either 'standard' for standard scaling or 'minmax'
        for min-max scaling.
    numeric_cols: list, optional (default=None)
        The list of column names to scale. If None, uses default list of columns.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with the scaled numeric columns.
    """
    # Set default columns for scaling if not provided
    if numeric_cols is None:
        numeric_cols = ['energy_100g', 'fat_100g', 'carbohydrates_100g', 'proteins_100g']
    
    # Replace missing values ​​with the median of each column
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Create the corresponding scaler
    if how == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    
    # Apply scaling to numeric columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Return DataFrame with scaled data
    return df
