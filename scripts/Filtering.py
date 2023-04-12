from data_loader import *
import pandas as pd

class Filtering:

    d_df        = None
    d_endswith  = ["_t","_datetime","_url"]
    d_wendswith = ["_tags"]

    def __init__(
                    self,
                    df        = d_df,
                    endswith  = d_endswith,
                    wendswith = d_wendswith
                ):

        # Init attributes
        self.df        = df
        self.endswith  = endswith
        self.wendswith = wendswith

    def get_dataset_len(self):
        return len(self.df.columns)
        
    # Functions to get numerical features
    def get_numerical_features(self):
        df_num = self.df.select_dtypes(exclude=["object"])
        print("\nNumerical features list :")
        for feature in df_num.columns.tolist(): print(feature)
        return df_num

    # Functions to get categorical features
    def get_categorical_features(self):
        df_cat = self.df.select_dtypes(include=["object"])
        print("\nCategorical features list :")
        for feature in df_cat.columns.tolist(): print(feature)
        return df_cat
    
    def get_features_endswith(self, endswith, invert=False):
        print("\nGet features endswith")
        ft_endswith = []
        ft_list     = self.df.columns
        for feature in ft_list.tolist():
            for end in endswith:
                if feature.endswith(end):
                    feature_wendswith = feature.removesuffix(end)
                    if invert and feature_wendswith in ft_list:
                        ft_endswith.append(feature_wendswith)
                    else:
                        ft_endswith.append(feature)
        df_endswith = self.df[ft_endswith]
        return df_endswith
    
    def drop_features(self, features):
        print("\nDrop features processing...")
        for feature in features:
            print("Deleting feature : "+feature)
        return self.df.drop(features, axis=1, inplace=True)
    
    # Function to preprocess the Dataframe
    def filtering(self):

        self.get_numerical_features()

        self.get_categorical_features()

        ft_endswith = self.get_features_endswith(endswith=self.endswith)
        # print(ft_endswith.head())

        self.drop_features(ft_endswith)

        ft_wendswith = self.get_features_endswith(endswith=self.wendswith, invert=True)
        # print(ft_wendswith.head())

        self.drop_features(ft_wendswith)

        # print(self.df[['ingredients_that_may_be_from_palm_oil_tags']].head(50))
    
        return self.df
        
if __name__ == "__main__":
    v_file_path = r"D:\Python_app\teaching_ml_2023/data/en.openfoodfacts.org.products.csv"
    v_nrows     = 10000

    # Execute filtering
    df_train = get_data(file_path=v_file_path, nrows=v_nrows)
    df_train = Filtering(df_train).filtering()
    print(df_train.head())
