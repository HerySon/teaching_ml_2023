""" The function selects the numeric columns of the given dataset. 
It then uses the chosen scaler to normalize the numeric columns and 
replaces the original dataset with the normalized columns."""

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
import pandas as pd
import numpy as np

# Scaler_type must be equal to "standard" ; "minmax" ; 'robust' or 'maxabs'
def normalize_dataset(dataset, scaler_type):
    # Select the numeric columns
    numeric_columns = dataset.select_dtypes(include=[np.number]).columns
    
    # Apply the chosen scaler to the numeric columns
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    elif scaler_type == 'maxabs':
        scaler = MaxAbsScaler()
    else:
        raise ValueError("Scaler type not recognized.")
    
    
    # Replace the numeric columns in the dataset with the normalized columns
    dataset[numeric_columns] = scaler.fit_transform(dataset[numeric_columns])
    
    return dataset