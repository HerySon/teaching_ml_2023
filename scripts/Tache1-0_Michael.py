import yaml
import pandas as pd
from sklearn.impute import KNNImputer

def clean_data(dataset, null_values_percent_authorized = 30, n_neighbors=5):
    df_null_val = dataset.isnull().sum() * 100 / len(dataset)
    good_parameter = df_null_val[df_null_val < null_values_percent_authorized]
    dataset = dataset[good_parameter.index]

    str_cols = dataset.select_dtypes(include=['object']).columns.tolist()
    dataset.loc[:, str_cols] = dataset[str_cols].fillna('?')

    imputer = KNNImputer(n_neighbors=n_neighbors)
    num_cols = dataset.select_dtypes(include=['int64','float64']).columns.tolist()
    dataset.loc[:, num_cols] = imputer.fit_transform(dataset[num_cols])

    return dataset

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
    data = get_data(file_path = "../data/en.openfoodfacts.org.products.csv", nrows=50)
    print(f"data set shape is {data.shape}") 
    data = clean_data(data, 35)