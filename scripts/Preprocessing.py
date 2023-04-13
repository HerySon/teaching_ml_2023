from data_loader import *
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class Preprocessing:
    """Class to preprocessing pandas dataframe
    Args:
        df (DataFrame): pandas dataframe
        percent (integer): percent of missing data
        num_imput (string): method to impute numerical features missing values
        cat_imput (string): method to impute categorical features missing values
    Returns:
        df (DataFrame): return preprocessed pandas dataframe
    @Author: Thomas PAYAN
    """

    def __init__(
                    self,
                    df                  = None,
                    percent             = 70,
                    num_imput           = 'mean',
                    cat_imput           = 'mode',
                    label_encode_method = 'label'
                ):

        self.df                  = df
        self.percent             = percent
        self.num_imput           = num_imput
        self.cat_imput           = cat_imput
        self.label_encode_method = label_encode_method

    def convert_numpy_to_pandas(self, np_array):
        """Convert numpy array to pandas Dataframe
        Args:
            np_array (array) : numpy array
        Returns:
            df (DataFrame): return pandas dataframe
        @Author: Thomas PAYAN
        """
        return pd.DataFrame(np_array)
        
    def count_duplicated_values(self):
        """Count duplicated values for each feature
        Returns:
            df (DataFrame): return sum of duplicated values
        @Author: Thomas PAYAN
        """
        return self.df.duplicated().sum()

    def drop_duplicated_values(self):
        """Drop duplicated values
        Returns:
            df (DataFrame): return dataframe without duplicated values
        @Author: Thomas PAYAN
        """
        print("\nRows number before drop duplicated values : %s" % len(self.df))
        if(self.count_duplicated_values() > 0):
            self.df.drop_duplicates(inplace=True)
        print("\nRows number after drop duplicated values : %s" % len(self.df))
        return self.df

    def display_missing_values(self): 
        """Display missing values
        @Author: Thomas PAYAN
        """
        for col in self.df.columns.tolist():          
            print("{} column missing values : {}".format(col, self.df[col].isnull().sum()))

    def drop_missing_values(self):
        """Drop missing values
        Returns:
            df (DataFrame): return dataframe without missing values
        @Author: Thomas PAYAN
        """
        self.display_missing_values()

        calc_null = [(col, self.df[col].isna().mean()*100) for col in self.df]
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

    def impute_numeric_features(self):
        """Impute numerical features missing values
        Returns:
            df (DataFrame): return dataframe with numerical features imputed
        @Author: Thomas PAYAN
        """
        print("\nPerforming numeric features imputation ("+self.num_imput+")")

        df_num = self.df.select_dtypes(include=["number"])

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
                print("\nWarning : select another method !")

        return self.df
                   
    def impute_categorical_features(self):
        """Impute categorical features missing values
        Returns:
            df (DataFrame): return dataframe with categorical features imputed
        @Author: Thomas PAYAN
        """
        print("\nPerforming categorical features imputation ("+self.cat_imput+")")

        df_cat = self.df.select_dtypes(include=["object"])

        match self.cat_imput:
            case 'mode':
                for col in df_cat.columns.tolist():
                    self.df[col].fillna(self.df[col].mode().iloc[0], inplace=True)
            case _:
                print("\nWarning : select another method !")

        return self.df
             
    def impute_missing_values(self):
        """Impute all features missing values
        Returns:
            df (DataFrame): return dataframe with all features imputed
        @Author: Thomas PAYAN
        """
        self.impute_numeric_features()
        self.impute_categorical_features()
        self.display_missing_values()
        return self.df

    def code_encoding(self):
        """Encode categorical features using category codes
        Returns:
            df (DataFrame): return dataframe with categorical features encoded
        @Author: Thomas PAYAN
        """
        print("\nCategory codes encoding")

        df_cat = self.df.select_dtypes(include=["object"])

        for col in df_cat.columns.tolist():
            self.df[col] = self.df[col].astype('category')
            self.df[col] = self.df[col].cat.codes

        return self.df
    
    def label_encoding(self):
        """Encode categorical features using LabelEncoder
        Returns:
            df (DataFrame): return dataframe with categorical features encoded
        @Author: Thomas PAYAN
        """
        print("\nLabel encoding")

        df_cat = self.df.select_dtypes(include=["object"])

        for cat in df_cat.columns.tolist():
            self.df[cat] = LabelEncoder().fit_transform(self.df[cat])

        return self.df
    
    def one_hot_encoding(self):
        """Encode categorical features using OneHotEncoder
        Returns:
            df (DataFrame): return dataframe with categorical features encoded
        @Author: Thomas PAYAN
        """
        print("\nOneHot encoding")

        encoded_df = OneHotEncoder(handle_unknown='ignore').fit_transform(self.df).toarray()
        self.df = self.convert_numpy_to_pandas(encoded_df)

        return self.df
    
    def categorical_features_encoding(self):
        """Categorical features encoding
        Returns:
            df (DataFrame): return new encoding dataframe
        @Author: Thomas PAYAN
        """
        print("\nPerforming categorical features encoding")

        match self.label_encode_method:
            case 'code':
                self.code_encoding()
            case 'label':
                self.label_encoding()
            case 'one_hot':
                self.one_hot_encoding()
            case _:
                print("\nWarning : select another method !")

        return self.df
    
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
    
    def get_feature_info(self, feature):
        """Get features information
        Args:
            feature (Dataframe) : feature to extract information
        Returns:
            info (DataFrame): return information
        @Author: Thomas PAYAN
        """
        info = None

        match self.num_imput:
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
    
    def correct_features_100g(self):
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

        Returns:
            df (DataFrame): return preprocessed dataframe
        @Author: Thomas PAYAN
        """
        excluded_features = ['energy-kj_100g','energy-kcal_100g','ph_100g','carbon-footprint_100g','nutrition-score-fr_100g','nutrition-score-uk_100g']
        df_100g           = self.get_features_endswith("_100g", excluded_features)

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
        
    def preprocessing(self):
        """Preprocess dataframe
        Returns:
            df (DataFrame): return preprocessed dataframe
        @Author: Thomas PAYAN
        """
        self.drop_duplicated_values()

        self.drop_missing_values()

        self.impute_missing_values()

        self.correct_features_100g()

        self.categorical_features_encoding()
    
        return self.df
        
if __name__ == "__main__":
    v_file_path = r"D:\Python_app\teaching_ml_2023/data/en.openfoodfacts.org.products.csv"
    v_nrows     = 10000

    # Execute preprocessing
    df_train = get_data(file_path=v_file_path, nrows=v_nrows)
    df_train = Preprocessing(df_train).preprocessing()
    print(df_train.head())