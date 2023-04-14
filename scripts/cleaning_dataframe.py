import pandas as pd
from data_loader import *
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class DataCleaner:
    def __init__(self, df, num_imput='mean', cat_encoding='label'):
        self.df = df
        self.num_imput = num_imput
        self.cat_encoding = cat_encoding
        
    def count_duplicated_values(self):
        """Compte les valeurs dupliquées pour chaque caractéristique
        Retourne :
            df (DataFrame) : retourne la somme des valeurs dupliquées """
        return self.df.duplicated().sum()
    
    def drop_duplicated_values(self):
        """Supprime les valeurs dupliquées
        Retourne :
            df (DataFrame) : retourne le dataframe sans les valeurs dupliquées """
        print("\nNombre de lignes avant la suppression des valeurs dupliquées : %s" % len(self.df))
        if(self.count_duplicated_values() > 0):
            self.df.drop_duplicates(inplace=True)
        print("\nNombre de lignes après la suppression des valeurs dupliquées : %s" % len(self.df))
        return self.df
    
    def impute_missing_values(self):
        """Remplit les valeurs manquantes pour les variables numériques
        Retourne :
            df (DataFrame) : retourne le dataframe avec les valeurs manquantes remplies """

        if self.num_imput == 'mean':
            self.df.fillna(self.df.mean(), inplace=True)
        elif self.num_imput == 'median':
            self.df.fillna(self.df.median(), inplace=True)
        elif self.num_imput == 'mode':
            self.df.fillna(self.df.mode().iloc[0], inplace=True)

        print("\nNombre de valeurs manquantes après l'imputation : %s" % self.df.isnull().sum().sum())
        return self.df

    def encode_categorical_variables(df, encoding_method='onehot'):
        """Encode categorical variables
        Args:
            df (DataFrame) : input dataframe
            encoding_method (str) : encoding method, either 'onehot' or 'label'
        Returns:
            df (DataFrame) : return encoded dataframe
        """
        categorical_cols = df.select_dtypes(include=['object']).columns
        if encoding_method == 'onehot':
            df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols, prefix_sep='_', drop_first=True)
        elif encoding_method == 'label':
            le = LabelEncoder()
            for col in categorical_cols:
                df[col] = le.fit_transform(df[col].astype(str))
        return df

    def cleaning(self):
        self.drop_duplicated_values()

        self.impute_missing_values()

        self.encode_categorical_variables()

if __name__ == "__main__":

    nrows = 10000
    file_path = "../data/en.openfoodfacts.org.products.csv"

    df_train = get_data(file_path, nrows)

    df_train = DataCleaner(df_train).cleaning()