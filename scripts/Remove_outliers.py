
import numpy as np


def remove_outliers(dataset,listofcolumns,n):
    """
This function takes a dataframe or a numpy array and removes outliers 
    using the median absolute deviation (MAD) method.
    
    Args:
        data: A pandas dataframe or numpy array containing the data.
        n: The number of standard deviations to use for outlier detection.
           usually the n=3 
        listofcolumns: Selected columns to apply the transformations on
    
    Returns:
        A dataframe or numpy array without the outliers.
    """
    #Calculate median of the data along each column
    median = np.median(dataset[listofcolumns], axis=0)
    
    # Calculate absolute deviation from median for each data point
    diff = np.abs(dataset[listofcolumns] - median)
    
    # Calculate median absolute deviation (MAD)
    mad = np.median(diff, axis=0)

    # Calculate threshold for outlier detection
    threshold = n * mad
    
    # Mask for identifying outlier values
    masked = np.abs(dataset[listofcolumns] - median) > threshold
    
    # Remove rows containing outliers
    data_cleaned = dataset[~masked.any(axis=1)]

    # Return the cleaned data
    return data_cleaned
