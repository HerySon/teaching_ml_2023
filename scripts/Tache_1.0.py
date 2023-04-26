import numpy as np
import pandas as pd

""" 
Args :
    dataset = Dataframe of your dataset
    deletion_threshold = Number in percentage of null values threshold
Returns :
    Cleaned dataset with all empty column and duplicates deleted
    and values formatted
"""
def data_cleaning(dataset, deletion_threshold):
    """Delete columns which have more or equal percentage of the deletion_threshold"""
    null_rate = ((dataset.isnull().sum() / dataset.shape[0])*100).reset_index()
    null_rate.columns = ['Nom_colonne','Taux_Null']
    partial_null_rate = null_rate[null_rate['Taux_Null'] >= 100-deletion_threshold]
    print("Nombre de colonnes vides supprimées : ",partial_null_rate.shape[0],"/",dataset.shape[1])
    cols_to_drop = partial_null_rate['Nom_colonne']
    dataset.drop(cols_to_drop, axis=1, inplace=True)
    """Replace unknown data to NaN"""
    dataset.replace('unknown', np.nan, inplace=True)
    """Drop all the duplicates rows from the dataset"""
    dataset.drop_duplicates(subset ="code", keep = 'last', inplace=True)
    """
    Seen 9 columns which have the same informations
    (countries,countries_en,coutries_tags,states,states_en,states_tags,created_datetime_t,last_modified_datetime_t)
    So I dropped the unnecessary columns
    """
    countries = dataset.filter(regex = 'countries')
    countries = list(countries)
    countries.remove('countries_en')
    states = dataset.filter(regex = 'states')
    states = list(states)
    states.remove('states_en')
    time = dataset.filter(regex = '_t')
    time = list(time)
    drop_redundancy = countries + states + time
    dataset.drop(drop_redundancy, axis = 1, inplace=True)
    """Lower all the data from string columns"""
    lower_column = ['product_name', 'countries_en', 'pnns_groups_1', 'pnns_groups_2', 'states_en']
    for col in lower_column:
        dataset[col] = dataset[col].str.lower()
    return dataset

if __name__ == "__main__":
    dataset_directory = "data\en.openfoodfacts.org.products.csv"
    displayed_rows = 10
    df = pd.read_csv(dataset_directory, nrows = displayed_rows, sep='\t', encoding='utf-8')
    data_cleaning(df, 30)