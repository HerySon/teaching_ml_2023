import yaml
import pandas as pd
from sklearn.impute import KNNImputer
from data_loader import get_data

"""
Tache 1.0 : 

clean_data(dataset, null_values_percent_authorized = 30, n_neighbors=5):

    This function cleans the dataframe passed as parameter:
    The arguments passed as parameters are:
    - dataset:
        which represents the dataset to be cleaned
    - null_values_percent_authorized:
        it represents the maximum percentage of null values allowed per column (it is set by default to 30%)
    - n_neighbors:
        which corresponds to the number of neighboring samples that will be used to apply the KNNImputer algorithm.

    Algorithmic specificity:
        - Part 1: Definition of the percentage of null value per column and selection of the columns corresponding to the selection criteria chosen by null_values_percent_authorized
        - Part 2: Processing of selected columns that contain values that are not int or float (Replacement by '?')
        - Part 3: Processing of numeric values and imputation of null values using the KNNImputer algorithm.
        
    Function return:
        - The function returns a dataframe after treatments
"""

def clean_data(dataset, null_values_percent_authorized = 30, n_neighbors=5):
    #Part 1: Selection of columns corresponding to the value defined by null_values_percent_authorized

    df_null_val = dataset.isnull().sum() * 100 / len(dataset)
    good_parameter = df_null_val[df_null_val < null_values_percent_authorized]
    dataset = dataset[good_parameter.index]

    #Part 2: Processing of columns containing object values and replacement of null values with '?'

    str_cols = dataset.select_dtypes(include=['object']).columns.tolist()
    dataset.loc[:, str_cols] = dataset[str_cols].fillna('?')

    #Part 3: Processing of columns containing int values and replacement of null values by imputation using KNNImputer

    imputer = KNNImputer(n_neighbors=n_neighbors)
    num_cols = dataset.select_dtypes(include=['int64','float64']).columns.tolist()
    dataset.loc[:, num_cols] = imputer.fit_transform(dataset[num_cols])

    return dataset

if __name__ == "__main__":
    data = get_data(file_path = "../data/en.openfoodfacts.org.products.csv", nrows=50)
    print(f"data set shape is {data.shape}") 
    data = clean_data(data, 35)