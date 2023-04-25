from data_loader import *
import pandas as pd

class Filtering:
    """Class to filtering pandas dataframe
    Args:
        df (DataFrame): pandas dataframe
        endswith (list): list of features ending with
        wendswith (list): list (invert) of features ending with
    Returns:
        df (DataFrame): return filtered pandas dataframe
    @Author: Thomas PAYAN
    """

    def __init__(
                    self,
                    df        = None,
                    endswith  = ["_t","_datetime","_url"],
                    wendswith = ["_tags"]
                ):

        self.df        = df
        self.endswith  = endswith
        self.wendswith = wendswith
        
    def get_numerical_features(self):
        """Get numerical features
        Returns:
            df_num (DataFrame): return numerical features pandas dataframe
        @Author: Thomas PAYAN
        """
        df_num = self.df.select_dtypes(include=["number"])
        print("\nNumerical features list :")
        for feature in df_num.columns.tolist(): print(feature)
        return df_num

    def get_categorical_features(self):
        """Get categorical features
        Returns:
            df_cat (DataFrame): return categorical features pandas dataframe
        @Author: Thomas PAYAN
        """
        df_cat = self.df.select_dtypes(include=["object"])
        print("\nCategorical features list :")
        for feature in df_cat.columns.tolist(): print(feature)
        return df_cat
    
    def get_features_endswith(self, endswith, invert=False):
        """Get features endswith method : 
            User can delete features that do not match selection afterwards
        Args:
            endswith (list) : list of features ending with
            invert (boolean) : to get features without the end of tag
        Returns:
            df_endswith (DataFrame): return selected features pandas dataframe
        @Author: Thomas PAYAN
        """
        print("\nGet features endswith")
        ft_endswith = []
        ft_list     = self.df.columns
        
        for feature in ft_list.tolist():
            for end in endswith:
                if feature.endswith(end): # Feature name ends with selected item
                    feature_wendswith = feature.removesuffix(end) # Feature name without suffix
                    if invert: # If invert is True : invert the original selection
                        if feature_wendswith in ft_list: # Feature name exists into Dataframe colums list
                            ft_endswith.append(feature_wendswith) # Push feature name into list
                    else:
                        ft_endswith.append(feature) # Push feature name into list

        df_endswith = self.df[ft_endswith] 
        return df_endswith
    
    def drop_features(self, features):
        """Drop features list
        Args:
            features (Dataframe) : features to drop
        Returns:
            df (DataFrame): return droped features pandas dataframe
        @Author: Thomas PAYAN
        """
        print("\nDrop features processing...")
        for feature in features:
            print("Deleting feature : "+feature)
        return self.df.drop(features, axis=1, inplace=True)
    
    def filtering(self):
        """Filter dataframe
        Returns:
            df (DataFrame): return filtered pandas dataframe
        @Author: Thomas PAYAN
        """
        self.get_numerical_features()

        self.get_categorical_features()

        ft_endswith = self.get_features_endswith(endswith=self.endswith)
        self.drop_features(ft_endswith)

        ft_wendswith = self.get_features_endswith(endswith=self.wendswith, invert=True)
        self.drop_features(ft_wendswith)
    
        return self.df
        
if __name__ == "__main__":
    v_file_path = r"D:\Python_app\teaching_ml_2023/data/en.openfoodfacts.org.products.csv"
    v_nrows     = 10000

    # Execute filtering
    df_train = get_data(file_path=v_file_path, nrows=v_nrows)
    df_train = Filtering(df_train).filtering()
    print(df_train.head())
