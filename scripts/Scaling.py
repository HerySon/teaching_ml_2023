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
    """Class to scaling pandas dataframe
    Args:
        df (DataFrame): pandas dataframe
        scaling_method (string): method to scale pandas dataframe
    Returns:
        df (DataFrame): return scaled pandas dataframe
    @Author: Thomas PAYAN
    """

    def __init__(
                    self,
                    df             = None,
                    scaling_method = 'standard'
                ):

        self.df             = df
        self.scaling_method = scaling_method

    def convert_numpy_to_pandas(self, np_array):
        """Convert numpy array to pandas Dataframe
        Args:
            np_array (array) : numpy array
        Returns:
            df (DataFrame): return pandas dataframe
        @Author: Thomas PAYAN
        """
        return pd.DataFrame(np_array)

    def convert_categorical_features_to_numeric(self):
        """Convert categorical features to numeric
        Returns:
            df (DataFrame): return dataframe with categorical features converted
        @Author: Thomas PAYAN
        """
        print("\nPerforming categorical features convertion")

        df_cat = self.df.select_dtypes(include=["object"])

        for col in df_cat.columns.tolist():
            self.df[col] = self.df[col].astype('category')
            self.df[col] = self.df[col].cat.codes

        return self.df

    def standard_scaler(self):
        """Scale dataframe features with StandardScaler transformation
        Returns:
            scaled_df (DataFrame): return new scaled dataframe
        @Author: Thomas PAYAN
        """
        print("\nStandard scaling")
        scaled_df = StandardScaler().fit_transform(self.df)
        scaled_df = self.convert_numpy_to_pandas(scaled_df)
        return scaled_df
    
    def min_max_scaler(self):
        """Scale dataframe features with MinMaxScaler transformation
        Returns:
            scaled_df (DataFrame): return new scaled dataframe
        @Author: Thomas PAYAN
        """
        print("\nMin-max scaling")
        scaled_df = MinMaxScaler().fit_transform(self.df)
        scaled_df = self.convert_numpy_to_pandas(scaled_df)
        return scaled_df
    
    def max_abs_scaler(self):
        """Scale dataframe features with MaxAbsScaler transformation
        Returns:
            scaled_df (DataFrame): return new scaled dataframe
        @Author: Thomas PAYAN
        """
        print("\nMax-abs scaling")
        scaled_df = MaxAbsScaler().fit_transform(self.df)
        scaled_df = self.convert_numpy_to_pandas(scaled_df)
        return scaled_df
    
    def robust_scaler(self, quantile_start=25, quantile_end=75):
        """Scale dataframe features with RobustScaler transformation
        Args:
            quantile_start (integer) : start quantile range
            quantile_end (integer) : end quantile range
        Returns:
            scaled_df (DataFrame): return new scaled dataframe
        @Author: Thomas PAYAN
        """
        print("\nRobust scaling")
        scaled_df = RobustScaler(quantile_range=(quantile_start, quantile_end)).fit_transform(self.df)
        scaled_df = self.convert_numpy_to_pandas(scaled_df)
        return scaled_df
    
    def power_transformation(self, method='yeo-johnson'):
        """Scale dataframe features with Power transformation
        Args:
            method (tuple : {'yeo-johnson', 'box-cox'}) : power transform method
        Returns:
            scaled_df (DataFrame): return new scaled dataframe
        @Author: Thomas PAYAN
        """
        print("\nPower transformation ("+method+")")
        scaled_df = PowerTransformer(method=method).fit_transform(self.df)
        scaled_df = self.convert_numpy_to_pandas(scaled_df)
        return scaled_df
    
    def quantile_transformation(self, n_quantiles=200, output_distribution='uniform'):
        """Scale dataframe features with Quantile transformation
        Args:
            n_quantiles (integer) : number of quantiles to be computed
            output_distribution (tuple : {'uniform', 'normal'}) : marginal distribution for the transformed data
        Returns:
            scaled_df (DataFrame): return new scaled dataframe
        @Author: Thomas PAYAN
        """
        print("\nQuantile transformation ("+output_distribution+")")
        scaled_df = QuantileTransformer(n_quantiles=n_quantiles, output_distribution=output_distribution).fit_transform(self.df)
        scaled_df = self.convert_numpy_to_pandas(scaled_df)
        return scaled_df
    
    def normalize_transformation(self):
        """Scale dataframe features with Normalizer transformation
        Returns:
            scaled_df (DataFrame): return new scaled dataframe
        @Author: Thomas PAYAN
        """
        print("\nNormalize transformation")
        scaled_df = Normalizer().fit_transform(self.df)
        scaled_df = self.convert_numpy_to_pandas(scaled_df)
        return scaled_df

    def scaling_features(self):
        """Scale dataframe features
        Returns:
            df (DataFrame): return scaled dataframe features
        @Author: Thomas PAYAN
        """
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
                print("\nWarning : select another method !")

    def scaling(self):
        """Scale dataframe
        Returns:
            df (DataFrame): return scaled dataframe
        @Author: Thomas PAYAN
        """
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
