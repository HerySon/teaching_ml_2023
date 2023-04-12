import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

def clean_dataset(df):
    # Drop any columns with more than 50% missing values
    df.dropna(thresh=df.shape[0]*0.5, axis=1, inplace=True)

    # Remove any leading or trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Convert all string columns to lowercase
    string_cols = df.select_dtypes(include='object').columns
    df[string_cols] = df[string_cols].apply(lambda x: x.str.lower())

    # Remove any duplicate rows
    df.drop_duplicates(inplace=True)

    # Convert any date columns to datetime format
    date_cols = df.select_dtypes(include='datetime').columns
    df[date_cols] = df[date_cols].apply(pd.to_datetime)

    # Remove any non-numeric characters from numeric columns
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].apply(lambda x: pd.to_numeric(x.astype(str).str.replace('[^0-9\.]+', ''), errors='coerce'))

    # Fill missing values with the mean or median of the column
    numeric_imputer = SimpleImputer(strategy='median')
    categorical_imputer = SimpleImputer(strategy='most_frequent')

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = categorical_imputer.fit_transform(df[[col]])
        else:
            df[col] = numeric_imputer.fit_transform(df[[col]])

    # Remove any outliers in numeric columns using the IQR method
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

    # Standardize the numeric columns using the standard scaler
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Encode any categorical columns using label encoding or one-hot encoding
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if len(df[col].unique()) == 2:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
        else:
            encoder = OneHotEncoder(handle_unknown='ignore')
            encoded_cols = pd.DataFrame(encoder.fit_transform(df[[col]]).toarray(), columns=[f'{col}_{category}' for category in encoder.categories_[0]])
            df = pd.concat([df, encoded_cols], axis=1)
            df.drop(col, axis=1, inplace=True)

    return df
#keep variable with suffixe _100g 
def search_componant(df, suffix='_100g'):
    #df: dataframe de départ
    # Suffixe de la variable
  componant = []
  for col in df.columns:
      if '_100g' in col: componant.append(col)
  df_subset_columns = df[componant]
  return df_subset_columns
#Suppresion des variables redondantes par exemple le cas des variables suffixées par _tags ou _en qui ne font que reprendre d'autres features traduites ou simplifiées.

def search_redundant_col(df):
  category_columns = ['categories','categories_tags','categories_en']
  redundant_columns = []
  df[df[category_columns].notnull().any(axis=1)][['product_name'] + category_columns].sample(5)
  for col in df.columns:
    if "_en" in col:
      en = col.replace('_en','')
      tags = col.replace('_en','_tags')
      print("{:<20} 'Sans suffixe' -> {} ; 'Suffixe _tags' -> {}".format(col,
                                                                        en in df.columns, tags in df.columns))
      if en in df.columns : 
        redundant_columns.append(en)
      if tags in df.columns : 
        redundant_columns.append(tags)
  
    if '_tags' in col:
      tags_2 = col.replace('_tags','')
      print("{:<20} 'Suffixe _tags' -> {} ;".format(tags_2, tags_2 in df.columns))
      if tags_2 in df.columns :
        redundant_columns.append(col)

  return redundant_columns
df.drop(search_redundant_col(df), axis=1, inplace=True)
#Autres opérations spécifiques
#Les dates également comportent une certaine redondance. Entre les timestamp et les dates au format "yyyy-mm-dd", il est nécessaire d'en éliminer :
df['created_datetime'] = pd.to_datetime(df['created_t'], unit='s')
df['last_modified_datetime'] = pd.to_datetime(df['last_modified_t'], unit='s')
df = df.drop(['created_t','last_modified_t'], axis=1)
df.head()
#Suppression tous les produits qui n'ont ni nom, ni catégorie 
df_cleaned = df[~((df.product_name.isnull()) 
                        & ((df.pnns_groups_1 == "unknown") 
                           | (df.main_category_en == "unknown")))]
#On supprime les lignes dont toutes les numerical_features sont à 0 ou nulles
df_cleaned = df_cleaned.loc[~((df_cleaned[numerical_features]==0) | (df_cleaned[numerical_features].isnull())).all(axis=1)]
#On supprime les lignes contenant des valeurs négatives et des max aberrants
df_cleaned = df_cleaned[~(df_cleaned[numerical_features] < 0).any(axis=1)]
df_cleaned = df_cleaned[~(df_cleaned[numerical_features].isin([999999,9999999])).any(axis=1)]
# supprimer les lignes dont au moins 1 des variables de nutriments est supérieur au seuil pour les variabes _100g
g_per_100g_features = ['proteins_100g','fat_100g','carbohydrates_100g','sugars_100g','salt_100g',
                       'sodium_100g','saturated-fat_100g','fiber_100g']
df_cleaned = df_cleaned[~(df_cleaned[g_per_100g_features] > 100).any(axis=1)]
#saturated-fat_100g < fat_100g, de même sodium_100g < salt_100g.On supprime les lignes qui ne remplissement pas es conditions
df_cleaned = df_cleaned[~((df_cleaned['saturated-fat_100g'] > df_cleaned['fat_100g']) 
                                | (df_cleaned['sodium_100g'] > df_cleaned['salt_100g']))]
#Nous allons donc supprimer toutes les lignes dont la variable energy_100g est supérieur à 3700 (ou 900 kcal/100g).
df_cleaned = df_cleaned[~((df_cleaned['energy_100g'] > 3700) 
                                | (df_cleaned['energy-kcal_100g'] > 900))]
