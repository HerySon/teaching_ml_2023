
def select_features(filename, threshold, chunksize):
    """
    Selects the columns in the dataset with less than the threshold percent of missing values, and returns the resulting
    dataset with only those columns. Also shows the number of selected columns, the list of the emptiest columns, and
    a heatmap to visualize the missingness of the selected columns.

    Parameters:
    filename (str): The filename of the dataset to process.
    threshold (float): The threshold percent of missing values to use for selecting the columns.
    chunksize (int): The size of each chunk to use when reading the dataset.

    Returns:
    missing_percentages : The overall percentage of missing values for each column
    selected_columns :Columns that have less than the threshold of missing values
    """

    # Get the number of initial columns
    num_cols = len(pd.read_csv(filename, nrows=1).columns)
    print(f"Number of initial columns: {num_cols}")

    # Initialize a dictionary to store the overall missing values for each column
    missing_counts = {col: 0 for col in pd.read_csv(filename, nrows=1).columns}

    # Iterate over the dataset in chunks to reduce memory usage
    for chunk in pd.read_csv(filename, chunksize=chunksize):
        # Calculate the percentage of missing values for each column in the current chunk
        missing_percentage = chunk.isnull().sum() / len(chunk) * 100

        # Update the overall missing values for each column by adding by overwriting at each iteration
        for col in missing_counts:
            missing_counts[col] += missing_percentage[col]

    # Calculate the overall percentage of missing values for each column
    missing_percentages = pd.Series(missing_counts) / len(pd.read_csv(filename)) * 100

    # Select the columns that have less than the threshold of missing values
    selected_columns = list(missing_percentages[missing_percentages <= threshold].index)


    # Print the number of selected columns
    print(f"Number of selected columns: {len(selected_columns)}")


    return missing_percentages,selected_columns