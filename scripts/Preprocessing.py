import pandas as pd
import numpy as np
from statistics import mode
from data_loader import *
from sklearn.impute import KNNImputer

class Preprocessing:

    d_df      = None
    d_percent = 70
    d_num_imput = 'mean'
    d_cat_imput = 'mode'

    def __init__(
                    self,
                    df        = d_df,
                    percent   = d_percent,
                    num_imput = d_num_imput,
                    cat_imput = d_cat_imput
                ):

        # Init attributes
        self.df        = df
        self.percent   = percent
        self.num_imput = num_imput
        self.cat_imput = cat_imput
        
    # Functions to count duplicated values
    def count_duplicated_values(self):
        return self.df.duplicated().sum()

    # Function to drop duplicated values
    def drop_duplicated_values(self):
        print("\nRows number before drop duplicated values : %s" % len(self.df))
        if(self.count_duplicated_values() > 0):
            self.df.drop_duplicates(inplace=True)
        print("\nRows number after drop duplicated values : %s" % len(self.df))
        return self.df

    # Functions to display missing values
    def display_missing_values(self): 
        for col in self.df.columns.tolist():          
            print("{} column missing values : {}".format(col, self.df[col].isnull().sum()))

    # Function to drop missing values
    def drop_missing_values(self):
        self.display_missing_values()

        calc_null = [(c, self.df[c].isna().mean()*100) for c in self.df]
        calc_null = pd.DataFrame(calc_null, columns=["Feature", "Percent NULL"])
        print(calc_null.sort_values("Percent NULL", ascending=False))

        calc_null = calc_null[calc_null["Percent NULL"] > self.percent]
        print(calc_null.sort_values("Percent NULL", ascending=False))
        print(calc_null.count())

        list_calc_null = list(calc_null["Feature"])
        print("\nNumber features before droping when %s percent of values are NULL : %s" % (self.percent, len(self.df.columns)))
        self.df.drop(list_calc_null, axis=1, inplace=True)
        print("\nNumber features after droping when %s percent of values are NULL : %s" % (self.percent, len(self.df.columns)))

        return self.df

    # Function to impute numeric features missing values
    def impute_numeric_features(self):
        print("\nPerforming numeric features imputation")

        df_num = self.df.select_dtypes(exclude=["object"])

        match self.num_imput:
            case 'mean':
                for col in df_num.columns.tolist():
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
            case 'median':
                for col in df_num.columns.tolist():
                    self.df[col].fillna(self.df[col].median(), inplace=True)
            case 'mode':
                for col in df_num.columns.tolist():
                    self.df[col].fillna(self.df[col].mode().iloc[0], inplace=True)
            case _:
                for col in df_num.columns.tolist():
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                   
        return df_num

    # Function to impute categorical features missing values
    def impute_categorical_features(self):
        print("\nPerforming categorical features imputation")

        df_cat = self.df.select_dtypes(include=["object"])

        match self.cat_imput:
            case 'mode':
                for col in df_cat.columns.tolist():
                    self.df[col].fillna(self.df[col].mode().iloc[0], inplace=True)
            case _:
                for col in df_cat.columns.tolist():
                    self.df[col].fillna(self.df[col].mode().iloc[0], inplace=True)
             
        return df_cat

    # Function to impute missing values
    def impute_missing_values(self):
        encode_df_num = self.impute_numeric_features()
        encode_df_cat = self.impute_categorical_features()
        final_df = pd.concat([encode_df_num, encode_df_cat], axis=1,sort=False)
        return final_df

    # Function to preprocess the Dataframe
    def preprocessing(self):
        
        self.drop_duplicated_values()

        self.drop_missing_values()

        self.impute_missing_values()

if __name__ == "__main__":
    v_file_path = r"D:\Python_app\teaching_ml_2023/data/en.openfoodfacts.org.products.csv"
    v_nrows     = 10000

    df_train    = get_data(file_path=v_file_path, nrows=v_nrows)
    df_train    = Preprocessing(df_train).preprocessing()
