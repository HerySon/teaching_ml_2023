"""Data Preparation
"""

import data_loader as dl
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

# Call the get_data() function
data = dl.get_data(file_path='../data/en.openfoodfacts.org.products.csv', nrows=1000)

### Function that delete the columns that have more than 30% nan values ( also i replaced the unkown values with nan)
def data_with_less_30_nan(data):
    #copying data
    data_copy = data.copy(deep=True)

    # Select columns with 30% or more NaN values
    nan_percentage = data_copy.isna().mean() * 100
    columns_to_drop = nan_percentage[nan_percentage > 30].index

    print("columns to drop before replacing the uknown values : "+ str(len(columns_to_drop)))

    data = data.replace('unknown', np.nan)

    nan_percentage = data.isna().mean() * 100
    columns_to_drop = nan_percentage[nan_percentage > 30].index
    print("columns to drop after replacing the uknown values : "+ str(len(columns_to_drop)))
    data = data.drop(columns=columns_to_drop)

    return data



# print(data.info())

# data_cleaned = data_with_less_30_nan(data) 

# print (data_cleaned.info())






### Imputing the missing values with KNNImputer 

def imput_missing_values(numerical_data):

    # Identify columns with missing values
    columns_with_missing_values = numerical_data.columns[numerical_data.isnull().sum() > 0]
    print(columns_with_missing_values)

    # Create a pipeline for preprocessing
    preprocessing_pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),  # KNNImputer for imputing missing values
    ])

    # Fit and transform the data using the preprocessing pipeline
    data_preprocessed = preprocessing_pipeline.fit_transform(numerical_data)

    # Convert the preprocessed data back to a DataFrame
    data_preprocessed = pd.DataFrame(data_preprocessed, columns=numerical_data.columns)
    return data_preprocessed





## lets take only the numerical columns (float,int..)
# numerical_columns = data_cleaned.select_dtypes(include=[np.number])

# # we want to create a new DataFrame with only the selected columns
# numerical_data = data[numerical_columns.columns]
# print(numerical_data.head())

# data_preprocessed  = imput_missing_values(numerical_data)
# print(data_preprocessed.head())


