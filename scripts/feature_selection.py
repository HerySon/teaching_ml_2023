from data_loader import get_data
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import VarianceThreshold

def ordinal_encoding(df): 
    """Encode categorical feats using OrdinalEncoder
    Args:
        df (DataFrame): Dataframe object
    Returns:
        dataframe (DataFrame): output dataframe
    @Author: Nicolas THAIZE
    """
    encoded_df = df.copy()
    ord_enc = OrdinalEncoder()
    categorical_features = encoded_df.select_dtypes('object').columns
    encoded_df[categorical_features] = ord_enc.fit_transform(encoded_df[categorical_features])
    return encoded_df

def drop_low_var_feats(df, auto_ordinal_encode = True, threshold = 0.1):
    """Drop features that have > (1 - threshold %) simmilar values
    Args:
        df (DataFrame): Dataframe object
        auto_ordinal_encode (boolean): If true, encode categorical features
            By default True
        threshold (float): variance threshold to drop features
    Returns:
        dataframe (DataFrame): output dataframe
    
    @Author: Nicolas THAIZE
    """
    temp_df = df.copy()
    if auto_ordinal_encode:
        temp_df = ordinal_encoding(temp_df)
    var_thr = VarianceThreshold(threshold = threshold)
    var_thr.fit(temp_df)
    var_thr.get_support()
    low_var_feats = [column for column in temp_df.columns if column not in temp_df.columns[var_thr.get_support()]]
    return df.drop(low_var_feats,axis=1)

if __name__ == "__main__":
    df = get_data(file_path = "./data/en.openfoodfacts.org.products.csv", nrows=50)
    feature_selected_df = drop_low_var_feats(df)
    
