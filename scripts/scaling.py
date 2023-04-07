from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, normalize
from data_loader import get_data
import numpy as np

def scale_fit_df_by_method(df, method, **kwargs):
    """Fit a specific scaler using dataframe. Scaler option can be passed through kwargs
    Args:
        df (DataFrame): dataframe
        method (string): Scaler to fit
        kwags (any): Every scaler methods options
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
            scaler = normalize()
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
    df = df.select_dtypes([np.number])
    df_cols = df.columns
    df[df_cols] = scaler.transform(df[df_cols])
    return df

if __name__ == "__main__":
    data = get_data(file_path = "./data/en.openfoodfacts.org.products.csv", nrows=50)
    scaler = scale_fit_df_by_method(data, "std", with_mean=False)
    result = scale_transform_df(data, scaler)
    print(result.shape)