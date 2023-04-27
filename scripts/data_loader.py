
"""Data loading tools
"""
import yaml
import pandas as pd
import data_cleaning

def read_config(file_path='config.yaml') -> None:
    """Reads configuration file
    Args:
        file_path (str, optional): file path
    Returns:
        dict: Parsed configuration file
    """
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def get_data(file_path=None, nrows=None, verbose=False) -> pd.DataFrame:
    """Loads data into a pandas DataFrame
    Args:
        file_path (str, optional): file path of dataset
            By default load data set from static web page
        nrows (int, optional): number or rows to loads from dataset
            By default loads all dataset  
        verbose (bool): whether to get verbose output    
    Returns:
        dataframe: output dataframe
    """
    if file_path is None:
        cfg = read_config()
        file_path = cfg['paths']['eng_dataset']
    if verbose:    
        print("Reading dataset ...")  
             
    return pd.read_csv(file_path,sep="\t", encoding="utf-8",
                       nrows=nrows, low_memory=False)



if __name__ == "__main__":
    data = get_data(file_path = "../data/en.openfoodfacts.org.products.csv", nrows=1000)
    print(f"data set shape is {data.shape}") 


