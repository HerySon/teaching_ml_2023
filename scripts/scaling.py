from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer
from data_loader import get_data
import numpy as np
from numpy.testing import assert_equal
import pandas as pd

def scale_data(dataset, how='standard', nrm = 'l2', qtl_rng = (25.0, 75.0)):
    """
    This function aim to standardize values in dataset. You pass the OpenFoodFact dataset as entry and you get the standardized dataset at the end.

    Args :
        dataset -- the dataset who contains OpenFoodFact data (required)
        
        how -- method used to scale data. 'norm' for Normalizer, 'standard' for StandardScaler, 'minmax' for MinMaxScaler, 'robust' for RobustScaler. 
        Default = 'standard'
        
        nrm -- The norm to use to normalize each non zero sample. If norm=’max’ is used, values will be rescaled by the maximum of the absolute values.
        Possibles values are : ['l1', 'l2', 'max']. Default = 'l2'
        
        qtl_rng -- tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0. 
        Quantile range used to calculate the interquartile range for each feature in the training set. Default = (25.0, 75.0)

    Returns :
        Dataset of OpenFoodFact with standardized values.
    @Author: Nans, Grace 
    """
    
    assert_equal(type(dataset), type(pd.DataFrame()), err_msg='Input is not Pandas Dataframe.', verbose=True)
    if how == 'standard':
        scaler = StandardScaler()
    elif how == 'norm':
        scaler = Normalizer(norm=nrm)
    elif how == 'minmax':
        scaler = MinMaxScaler()
    elif how == 'robust':
        scaler = RobustScaler(quantile_range=qtl_rng)
    vars_to_scale = dataset.select_dtypes(include=['number']).columns
    scaler.fit(dataset[vars_to_scale])
    dataset[vars_to_scale] = scaler.transform(X[vars_to_scale])
    return dataset
    
def scale_fit_df_by_method(df, method, **kwargs):
    """Fit a specific scaler using dataframe. Scaler option can be passed through kwargs
    Args:
        df (DataFrame): dataframe
        method (string): Scaler to fit
        kwargs (any): Every scaler methods options
    Returns:
        scaler: fitted scaler object
    @Author: Nicolas THAIZE
    """
    df = df.select_dtypes([np.number])
    match method:
        case "std":
            with_mean = kwargs.get('with_mean', True)
            with_std = kwargs.get('with_std', True)
            scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
        case "minmax":
            feature_range = kwargs.get('feature_range', (0,1))
            scaler = MinMaxScaler(feature_range=feature_range)
        case "minabs":
            scaler = MaxAbsScaler()
        case "robust":
            with_centering = kwargs.get('with_centering', True)
            with_scaling = kwargs.get('with_scaling', True)
            quantile_range = kwargs.get('quantile_range', (25.0, 75.0))
            unit_variance = kwargs.get('unit_variance', False)
            scaler = RobustScaler(with_centering=with_centering, with_scaling=with_scaling, quantile_range=quantile_range, unit_variance=unit_variance)
        case "normalize":
            norm = kwargs.get('norm', "l2")
            scaler = Normalizer(norm=norm)
        case _:
            raise ValueError
        
    df_cols = df.columns
    return scaler.fit(df[df_cols])
    

def scale_transform_df(df, scaler):
    """Transform dataset with provided scaler
    Args:
        df (DataFrame): dataframe
        scaler (Scaler): fitted scaler to use 
    Returns:
        dataframe: output scaled dataframe
    @Author: Nicolas THAIZE
    """
    df_cols = df.select_dtypes([np.number]).columns
    df[df_cols] = scaler.transform(df[df_cols])
    return df

if __name__ == "__main__":
    data = get_data(file_path = "../data/en.openfoodfacts.org.products.csv", nrows=50)
    scaler = scale_fit_df_by_method(data, "std", with_mean=False)
    result = scale_transform_df(data, scaler)
    print(result.shape)

