from sklearn.preprocessing import StandardScaler
import numpy as np

def get_num_feats(df):
    """Return only numerical features
    Args:
        df (DataFrame, optional): pandas dataframe
    Returns:
        dataframe: output only numerical data from dataframe
    @Author: Nicolas THAIZE
    """
    return df.select_dtypes([np.number])

def std_scale_df(df):
    """Standard dataframe data scaling
    Args:
        dataframe (DataFrame, optional): pandas dataframe 
    Returns:
        dataframe: output scaled dataframe
    @Author: Nicolas THAIZE
    """
    temp_df = df.copy()
    temp_df_num_cols = get_num_feats(temp_df).columns
    scaler = StandardScaler()
    temp_df[temp_df_num_cols] = scaler.fit_transform(temp_df[temp_df_num_cols])
    return temp_df

def drop_cols_fullna(df):
    """Drop na filled columns
    Args:
        df (DataFrame, optional): pandas dataframe  
    Returns:
        dataframe: output cleaned dataframe
    @Author: Nicolas THAIZE
    """
    return df.dropna(axis=1, how='all')

def impute_missing(df):
    """Impute data
    Args:
        df (DataFrame, optional): pandas dataframe  
    Returns:
        dataframe: output imputed dataframe
    @Author: Nicolas THAIZE
    """
    temp_df = df.copy()
    temp_df = drop_cols_fullna(temp_df)
    return temp_df.fillna(temp_df.mean())
