
"""Data loading tools
"""
import yaml
import pandas as pd
import data_cleaning

def read_config(file_path='./config.yaml'):
    """Reads configuration file
    Args:
        file_path (str, optional): file path
    Returns:
        dict: Parsed configuration file
    """
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def get_data(file_path=None, nrows=None):
    """Loads data
    Args:
        file_path (str, optional): file path of dataset
            By default load data set from static web page
        nrows (int, optional): number or rows to loads from dataset
            By default loads all dataset  
    Returns:
        dataframe: output dataframe
    """
    if file_path is None:
        cfg = read_config()
        file_path = cfg['paths']['eng_dataset']
    print("Reading dataset ...")    
    return pd.read_csv(file_path,sep="\t", encoding="utf-8",
                       nrows=nrows, low_memory=False)



if __name__ == "__main__":
    data = get_data(file_path = "../data/en.openfoodfacts.org.products.csv", nrows=200)
    print(f"data set shape is {data.shape}")
    #data_cleaning.print_columns(data)
    # Pass the column names as a list
    #data_cleaning.distplot2x2(data, ['energy_100g','fat_100g','saturated-fat_100g','trans-fat_100g'])
    #data_cleaning.print_unique_values(data["countries"])
    #print(data_cleaning.parse_countries(data["countries"]))

    df = data_cleaning.create_test_dataframe()
    print(df)
    new_dataset = data_cleaning.replace_country_name_with_code(df, "countries")
    print(new_dataset)





