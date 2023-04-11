import yaml
import pandas as pd

def outliers_gestion(dataset):
    columns = dataset.select_dtypes(include=['int64','float64']).columns.tolist()

    dataset_copy = dataset.copy()
    for col in columns:
        q1 = dataset_copy[col].quantile(0.25)
        q3 = dataset_copy[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers_indices = dataset_copy[(dataset_copy[col] < lower_bound) | (dataset_copy[col] > upper_bound)].index
        median = dataset_copy[col].median()
        dataset_copy.loc[outliers_indices, col] = median

    return dataset_copy

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
    data_res = outliers_gestion(data)