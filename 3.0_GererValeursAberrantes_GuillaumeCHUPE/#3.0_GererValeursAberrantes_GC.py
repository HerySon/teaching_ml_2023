def remove_negative_values(data, columns):
    """
    Removes negative values from specified columns of the dataframe.

    Args:
        data (pandas.DataFrame): Dataframe to modify.
        columns (list): List of column names to modify.

    Returns:
        pandas.DataFrame: Modified dataframe.
    """
    for col in columns:
        data = data[data[col] >= 0]
    return data

def remove_outliers_tukey(data, columns, factor=1.5):
    """
    Removes outliers from specified columns of the dataframe using the Tukey method.
    Q1-1.5IQR | Q3+1.5IQR

    Args:
        data (pandas.DataFrame): Dataframe to modify.
        columns (list): List of column names to modify.
        factor (float): Factor to calculate the Tukey range (default 1.5).

    Returns:
        pandas.DataFrame: Modified dataframe.
    """
    for col in columns:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    return data

def remove_outliers_zscore(data, columns, threshold=3):
    """
    Removes outliers from specified columns of the dataframe using the Z-score method.
    Z-score = (value - mean) / standard deviation

    Args:
        data (pandas.DataFrame): Dataframe to modify.
        columns (list): List of column names to modify.
        threshold (float): Threshold for the Z-score (default 3).

    Returns:
        pandas.DataFrame: Modified dataframe.
    """
    for col in columns:
        z_scores = (data[col] - data[col].mean()) / data[col].std()
        data = data[abs(z_scores) <= threshold]
    return data

from sklearn.ensemble import IsolationForest

def remove_outliers_iforest(data, columns, contamination=0.05):
    """
    Removes outliers from specified columns of the dataframe using the IsolationForest method.

    Args:
        data (pandas.DataFrame): Dataframe to modify.
        columns (list): List of column names to modify.
        contamination (float): Proportion of outliers expected in the dataset (default 0.05).

    Returns:
        pandas.DataFrame: Modified dataframe.
    """
    clf = IsolationForest(contamination=contamination, random_state=42)
    clf.fit(data[columns])
    y_pred = clf.predict(data[columns])
    data = data[y_pred == 1]
    return data

