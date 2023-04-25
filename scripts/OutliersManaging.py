from data_loader import *
import pandas as pd

class OutliersManaging:
    """Class to managing outliers
    Args:
        df (DataFrame): pandas dataframe
    Returns:
        df (DataFrame): return managed pandas dataframe
    @Author: Thomas PAYAN
    """

    def __init__(
                    self,
                    df = None
                ):

        self.df = df

    def get_features_endswith(self, endswith, ft_exclude=[]):
        """Get features endswith method
        Args:
            endswith (string|tuple) : list of features ending with
            ft_exclude (array) : list of features to exclude
        Returns:
            df_endswith (DataFrame): return selected features pandas dataframe
        @Author: Thomas PAYAN
        """
        print("\nGet features endswith")
        ft_endswith = []
        ft_list     = self.df.columns
        
        for feature in ft_list.tolist():
            if feature.endswith(endswith):
                ft_endswith.append(feature)

        res = filter(lambda i: i not in ft_exclude, ft_endswith)
        df_endswith = self.df[res]
        return df_endswith
    
    def get_feature_info(self, feature, method='mean'):
        """Get features information
        Args:
            feature (pandas Series) : feature to extract information
        Returns:
            info (string|integer): return information
        @Author: Thomas PAYAN
        """
        info = None

        match method:
            case 'mean':
                print('mean')
                info = feature.mean()
            case 'median':
                info = feature.median()
            case 'mode':
                info = feature.mode().iloc[0]
            case _:
                print("\nWarning : select another method !")

        print(info)
        return info
    
    def correct_features_100g(self, ft_exclude=[]):
        """Features correction ending with "_100g" (keep features expressed in percentage/grams)
            - Check feature : 0 <= min and max <= 100 and 0 <= mean <= 100
            - Perform an imputation of erroneous data (mean)
        
        Exclude the following variables :
            - energy-kj_100g          : unit in kj
            - energy-kcal_100g        : unit in kcal
            - ph_100g                 : unit in ph
            - carbon-footprint_100g   : emprunte carbon 
            - nutrition-score-fr_100g : fr nutrition score
            - nutrition-score-uk_100g : uk nutrition score

        Args:
            ft_exclude (list) : list of excluded features
        Returns:
            df (DataFrame): return preprocessed dataframe
        @Author: Thomas PAYAN
        """
        df_100g = self.get_features_endswith("_100g", ft_exclude)

        for feature in df_100g.columns.tolist():
            infos = self.df[feature].describe().loc[['min','mean','max']]
            if not pd.isna(infos['min']):
                if infos['min'] <= 0 or 100 <= infos['max']:
                    if 0 <= infos['mean'] <= 100:
                        print("Mean OK ! Probably some wrong values : min|max\nImpute processing...")
                        correct_value = self.get_feature_info(self.df[feature])
                        self.df.loc[(self.df[feature] < 0) | (100 < self.df[feature]), feature] = correct_value
                    else:
                        print("Mean KO ! Probably some wrong values : min|max|mean\nDon't process imputing...")
                else:
                    print("Good values")
            else:
                print("NaN values")

        return self.df
        
    def outliers_managing(self):
        """Managing outliers in pandas dataframe
        Returns:
            df (DataFrame): return managed dataframe
        @Author: Thomas PAYAN
        """
        ft_exclude = [
                        'energy-kj_100g',
                        'energy-kcal_100g',
                        'ph_100g',
                        'carbon-footprint_100g',
                        'nutrition-score-fr_100g',
                        'nutrition-score-uk_100g'
                    ]
        self.df = self.correct_features_100g(ft_exclude)
    
        return self.df
        
if __name__ == "__main__":
    v_file_path = r"D:\Python_app\teaching_ml_2023/data/en.openfoodfacts.org.products.csv"
    v_nrows     = 10000

    # Execute outliers managing
    df_train = get_data(file_path=v_file_path, nrows=v_nrows)
    df_train = OutliersManaging(df_train).outliers_managing()
    print(df_train.head())