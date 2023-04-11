from data_loader import *
# from Preprocessing import *
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

class Scaling:

    d_df             = None
    d_scaling_method = 'standard'

    def __init__(
                    self,
                    df             = d_df,
                    scaling_method = d_scaling_method
                ):

        # Init attributes
        self.df             = df
        self.scaling_method = scaling_method

    # Function to convert numpy array to pandas Dataframe
    def convert_numpy_to_pandas(self, df):
        return pd.DataFrame(df)

    # Function to convert categorical features to numeric
    def convert_categorical_features_to_numeric(self):
        print("\nPerforming categorical features convertion")

        df_cat = self.df.select_dtypes(include=["object"])

        for col in df_cat.columns.tolist():
            self.df[col] = self.df[col].astype('category')
            self.df[col] = self.df[col].cat.codes

    # Function to Standard scaling
    def standard_scaler(self):
        print("\nStandard scaling")
        scaled_df = StandardScaler().fit_transform(self.df)
        scaled_df = self.convert_numpy_to_pandas(scaled_df)
        return scaled_df
    
    # Function to Min-max scaling
    def min_max_scaler(self):
        print("\nMin-max scaling")
        scaled_df = MinMaxScaler().fit_transform(self.df)
        scaled_df = self.convert_numpy_to_pandas(scaled_df)
        return scaled_df
    
    # Function to Max-abs scaling
    def max_abs_scaler(self):
        print("\nMax-abs scaling")
        scaled_df = MaxAbsScaler().fit_transform(self.df)
        scaled_df = self.convert_numpy_to_pandas(scaled_df)
        return scaled_df
    
    # Function to Robust scaling
    def robust_scaler(self, v_quantile_start=25, v_quantile_end=75):
        print("\nRobust scaling")
        scaled_df = RobustScaler(quantile_range=(v_quantile_start, v_quantile_end)).fit_transform(self.df)
        scaled_df = self.convert_numpy_to_pandas(scaled_df)
        return scaled_df
    
    # Function to Power transformation
    def power_transformation(self, v_method='yeo-johnson'): # box-cox
        print("\nPower transformation ("+v_method+")")
        scaled_df = PowerTransformer(method=v_method).fit_transform(self.df)
        scaled_df = self.convert_numpy_to_pandas(scaled_df)
        return scaled_df
    
    # Function to Quantile transformation
    def quantile_transformation(self, v_output_distribution='uniform'): # normal
        print("\nQuantile transformation ("+v_output_distribution+")")
        scaled_df = QuantileTransformer(output_distribution=v_output_distribution).fit_transform(self.df)
        scaled_df = self.convert_numpy_to_pandas(scaled_df)
        return scaled_df
    
    # Function to Normalize transformation
    def normalize_transformation(self):
        print("\nNormalize transformation")
        scaled_df = Normalizer().fit_transform(self.df)
        scaled_df = self.convert_numpy_to_pandas(scaled_df)
        return scaled_df

    # Function to scale features
    def scaling_features(self):
        print("\nPerforming features scaling")

        match self.scaling_method:
            case 'standard':
                self.df = self.standard_scaler()
            case 'min_max':
                self.df = self.min_max_scaler()
            case 'max_abs':
                self.df = self.max_abs_scaler()
            case 'robust':
                self.df = self.robust_scaler()
            case 'power':
                self.df = self.power_transformation()
            case 'quantile':
                self.df = self.quantile_transformation()
            case 'normalize':
                self.df = self.normalize_transformation()
            case _:
                self.df = self.standard_scaler()

    # Function to scale Dataframe features
    def scaling(self):
        
        self.convert_categorical_features_to_numeric()

        self.scaling_features()

        return self.df

if __name__ == "__main__":
    v_file_path = r"D:\Python_app\teaching_ml_2023/data/en.openfoodfacts.org.products.csv"
    v_nrows     = 10000

    # Execute scaling
    df_train = get_data(file_path=v_file_path, nrows=v_nrows)
    # df_train = Preprocessing(df_train).preprocessing()
    # print(df_train.head())
    
    df_train = Scaling(df_train).scaling()
    print(df_train.head())
