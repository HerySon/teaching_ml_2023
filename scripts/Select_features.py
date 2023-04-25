import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def select_features(filepath, threshold, chunksize):
    """
    Selects the columns in the dataset with less than the threshold percent of missing values, and returns the resulting
    dataset with only those columns. Also shows the number of selected columns, the list of the emptiest columns, and
    a heatmap to visualize the missingness of the selected columns.

    Parameters:
    filepath (str): The filepath of the dataset to process.
    threshold (float): The threshold percent of missing values to use for selecting the columns.
    chunksize (int): The size of each chunk to use when reading the dataset.

    Returns:
    selected_df (pandas.DataFrame): The resulting dataset with only the selected columns.
    """

    # Get the number of initial columns
    num_cols = len(pd.read_csv(filepath,header="infer" ,nrows=1).columns)
    print(f"Number of initial columns: {num_cols}")

    # Initialize a dictionary to store the overall missing values for each column
    missing_counts = {col: 0 for col in pd.read_csv(filepath,header="infer" , nrows=1).columns}

    datalenght= 0
    # Iterate over the dataset in chunks to reduce memory usage
    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        # Calculate the percentage of missing values for each column in the current chunk
        missing = chunk.isnull().sum()
        datalenght += len(chunk)
        
        # Update the overall missing values for each column by adding by overwriting at each iteration
        for col in missing_counts:
            missing_counts[col] += missing[col]

    # Calculate the overall percentage of missing values for each column
    missing_percentages = pd.Series(missing_counts) / datalenght *100

    # Select the columns that have less than the threshold of missing values
    selected_columns = list(missing_percentages[missing_percentages <= threshold].index)

    # Print the number of selected columns
    print(f"Number of selected columns: {len(selected_columns)}")


    return missing_percentages,selected_columns


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Select features in a dataset based on missing values.")
    parser.add_argument("filepath", type=str, help="The filepath of the datasetin csv with sep as tabulation to process.")
    parser.add_argument("threshold", type=float, help="The threshold percent of missing values to use for selecting the columns.")
    parser.add_argument("chunksize", type=int, help="The size of each chunk to use when reading the dataset.")
    args = parser.parse_args()

    # Run the select_features function with the provided arguments
    select_features(args.filepath, args.threshold, args.chunksize)


if __name__ == "__main__":
    main()
