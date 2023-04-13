import pandas as pd
import numpy as np

""" 
Args :
    df = Dataframe of your dataset
    cat_threshold = Defines the maximum number of categories for a column to be considered nominal
Operating :
1)Identifying the numeric columns
2)Identifies the categorical
3)Separates the categorical columns into two lists: ordinal_cols for ordinal and nominal_cols for nominal columns
4)Downcastes the digital columns to save memory space.
5)Selects the columns corresponding to numerical, ordinal and nominal categories and a maximum number of 20 nominal category columns using the cat_threshold criterion.
6)Returns a new df_selected DataFrame containing only the selected columns.
"""

def filter_columns(df, cat_threshold=50):
    """List of numeric columns"""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    """List of categorical columns"""
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Sélection des colonnes catégorielles ordinales et non-ordinales
    ordinal_cols = []
    nominal_cols = []
    for col in categorical_cols:
        if len(df[col].unique()) <= 10:
            ordinal_cols.append(col)
        elif len(df[col].unique()) <= cat_threshold:
            nominal_cols.append(col)

    """Downcasting of numerical columns"""
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], downcast='float')

    """Criteria for filtering categorical variables according to their number of categories"""
    if len(nominal_cols) > 20:
        nominal_cols = nominal_cols[:20]

    selected_cols = numeric_cols + ordinal_cols + nominal_cols
    df_selected = df[selected_cols].copy()

    return df_selected

if __name__ == "__main__":
    dataset_directory = "data\en.openfoodfacts.org.products.csv"
    displayed_rows = 100
    df = pd.read_csv(dataset_directory, nrows = displayed_rows, sep='\t', encoding='utf-8')
    filter_columns(df,20)