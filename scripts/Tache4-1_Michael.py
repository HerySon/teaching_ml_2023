import seaborn as sns
import pandas as pd

def multi_var_viz(dataset, columns=[]):
    num_cols = data.select_dtypes(include=['int64','float64']).columns.tolist()
    data_int = data.loc[:, num_cols]
    corr = data_int.corr()
    sns.heatmap(corr, cmap='coolwarm')
    sns.pairplot(data_int, vars=columns)

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
    multi_var_viz(data, ["energy_100g", "completeness", "fat_100g"])
