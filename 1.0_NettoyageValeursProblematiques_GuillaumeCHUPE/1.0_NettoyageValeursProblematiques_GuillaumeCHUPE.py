#1.0_NettoyageValeursProblematiques_GuillaumeCHUPE

def drop_columns_with_missing_values(data, threshold=0.3):
    """
    Drops columns that have more than the specified threshold of missing values.

    Parameters:
    data (DataFrame): the dataset to clean
    threshold (float): the threshold above which a column will be dropped. Default is 0.3

    Returns:
    DataFrame: the cleaned dataset
    """
    # Compute the percentage of missing values for each column
    missing_percentage = data.isnull().sum() / data.shape[0]

    # Select columns to drop
    cols_to_drop = missing_percentage[missing_percentage > threshold].index

    # Drop the selected columns
    cleaned_data = data.drop(columns=cols_to_drop)

    return cleaned_data


def impute_missing_values(data):
    """
    Imputes missing values using the K-Nearest Neighbor algorithm with default settings.

    Parameters:
    data (DataFrame): the dataset to clean

    Returns:
    DataFrame: the cleaned dataset
    """
    # Create an instance of the KNNImputer class
    imputer = KNNImputer()

    # Impute missing values
    imputed_data = imputer.fit_transform(data)

    # Convert the numpy array back to a DataFrame
    cleaned_data = pd.DataFrame(imputed_data, columns=data.columns)

    return cleaned_data
