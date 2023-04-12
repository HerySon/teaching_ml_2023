import pandas as pd
import numpy as np

#Returns the columns that have more missing values than the treshold
def get_columns_matching_missing_values(dataframe, percentage_missing_values=None):
    if percentage_missing_values is None:
        percentage_missing_values = 0.5
    missing_values = dataframe.isnull().sum() / len(dataframe)
    #Above threshold
    matching_columns = missing_values[missing_values > percentage_missing_values].index.tolist()
    return matching_columns


def get_type_of_columns(dataframe, type):
    if type == 'num':
        matching_columns = dataframe.select_dtypes(include=np.number).columns.tolist()
    elif type == 'cat_ord' or type == 'cat_non_ord':
        matching_columns = dataframe.select_dtypes(include=['object', 'category']).columns.tolist()
    else:
        raise ValueError("Invalid type: must be 'num', 'cat_ord', or 'cat_non_ord'")
    return matching_columns

#Return the columns that have more categories than the value selected
def get_columns_with_max_cat(dataframe, max_number_of_cat=None):
    if max_number_of_cat is None:
        max_number_of_cat = 3
    cat_columns = dataframe.select_dtypes(include=['object', 'category']).columns
    matching_columns = [col for col in cat_columns if dataframe[col].nunique() <= max_number_of_cat]
    return matching_columns

#Delete lsit of columns from a dataframe
def delete_list_from_dataframe(dataframe, list_of_columns):
    dataframe_cleaned = dataframe.drop(columns=list_of_columns)
    return dataframe_cleaned


def main(dataframe):
    #Missing values removal
    percentage_missing_values = 0.9
    columns_to_delete_missing = get_columns_matching_missing_values(dataframe, percentage_missing_values)
    print(f"Deleting {len(columns_to_delete_missing)} columns due to missing values.")
    confirm = input("Do you want to proceed? (yes/no): ")
    if confirm.lower() == "yes":
        #Transformation to apply to the matching columns
        dataframe = delete_list_from_dataframe(dataframe, columns_to_delete_missing)

    #Type of data removal
    type_of_data = 'num'  # or 'cat_ord', 'cat_non_ord'
    columns_to_delete_type = get_type_of_columns(dataframe, type_of_data)
    print(f"Deleting {len(columns_to_delete_type)} columns of type '{type_of_data}'.")
    confirm = input("Do you want to proceed? (yes/no): ")
    if confirm.lower() == "yes":
        # Transformation to apply to the matching columns
        dataframe = delete_list_from_dataframe(dataframe, columns_to_delete_type)

    #Removal of columns with more cat than selected
    number_max_of_cat = 3
    columns_to_delete_max_cat = get_columns_with_max_cat(dataframe, number_max_of_cat)
    print(f"Deleting {len(columns_to_delete_max_cat)} columns with a maximum of {number_max_of_cat} categories.")
    confirm = input("Do you want to proceed? (yes/no): ")
    if confirm.lower() == "yes":
        # Transformation to apply to the matching columns
        dataframe_cleaned = delete_list_from_dataframe(dataframe, columns_to_delete_max_cat)
    else :
        dataframe_cleaned = dataframe


    return dataframe_cleaned

