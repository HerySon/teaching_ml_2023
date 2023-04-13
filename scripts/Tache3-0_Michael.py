import yaml
import pandas as pd
from data_loader import get_data

"""
Tache 3.0:

outliers_gestion(dataset, cols=None):

    This function cleans outliers on the dataframe passed as parameter:
    The arguments passed as parameters are:
    - dataset:
        which represents the dataset to be cleaned
    - cols :
        contains all columns where you want the treatment

    Algorithmic specificity:
        - Part 1: Selection of int and float columns to allow outlier calculations to manage possible errors
        - Part 2: Processing for columns that contain values measured on 100g and are not energy values
        - Part 2bis: Processing of other types of columns

    Function return:
        - The function returns a dataframe after treatments
"""

def outliers_gestion(dataset, cols=None):
    # Part 1: selection of columns

    if cols is None:
        columns = dataset.select_dtypes(include=['int64', 'float64']).columns.tolist()
    else:
        columns = dataset[cols].select_dtypes(include=['int64', 'float64']).columns.tolist()

    dataset_copy = dataset.copy()
    for col in columns:
        # Part 2: processing for columns that contain values measured on 100g and are not energy values

        if '100g' in col and 'energy' not in col:
            outliers_index = dataset_copy[dataset_copy[col] > 100].index
            median = dataset_copy[col].median()
            dataset_copy.loc[outliers_index, col] = median

        # Part 2bis: processing of other types of columns
        else:
            q1 = dataset_copy[col].quantile(0.25)
            q3 = dataset_copy[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers_index = dataset_copy[(dataset_copy[col] < lower_bound) | (dataset_copy[col] > upper_bound)].index
            median = dataset_copy[col].median()
            dataset_copy.loc[outliers_index, col] = median
    return dataset_copy

if __name__ == "__main__":
    data = get_data(file_path = "../data/en.openfoodfacts.org.products.csv", nrows=50)
    print(f"data set shape is {data.shape}") 
    data = outliers_gestion(data)