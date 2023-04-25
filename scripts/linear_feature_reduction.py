import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from data_loader import get_data
import seaborn as sns

def get_num_feats(df):
    """Return only numerical features
    Args:
        df (DataFrame, optional): pandas dataframe
    Returns:
        dataframe: output only numerical data from dataframe
    @Author: Nicolas THAIZE
    """
    return df.select_dtypes([np.number])

def std_scale_df(df):
    """Standard dataframe data scaling
    Args:
        dataframe (DataFrame, optional): pandas dataframe 
    Returns:
        dataframe: output scaled dataframe
    @Author: Nicolas THAIZE
    """
    temp_df = df.copy()
    temp_df_num_cols = get_num_feats(temp_df).columns
    scaler = StandardScaler()
    temp_df[temp_df_num_cols] = scaler.fit_transform(temp_df[temp_df_num_cols])
    return temp_df

def drop_cols_fullna(df):
    """Drop na filled columns
    Args:
        df (DataFrame, optional): pandas dataframe  
    Returns:
        dataframe: output cleaned dataframe
    @Author: Nicolas THAIZE
    """
    return df.dropna(axis=1, how='all')

def impute_missing(df):
    """Impute data
    Args:
        df (DataFrame, optional): pandas dataframe  
    Returns:
        dataframe: output imputed dataframe
    @Author: Nicolas THAIZE
    """
    temp_df = df.copy()
    temp_df = get_num_feats(temp_df)
    temp_df = drop_cols_fullna(temp_df)
    return temp_df.fillna(temp_df.mean())

def pca_fit_df(df, n_components=False):
    """Fit a PCA on df
    Args:
        df (DataFrame, optional): pandas dataframe  
        n_components (Int | float | str | None, optional) : number of components to create with pca
    Returns:
        pca: output pca object
    @Author: Nicolas THAIZE
    """
    num_df = get_num_feats(df) 
    if not n_components:
        n_components = len(num_df.columns)
    pca = PCA(n_components=n_components)
    pca.fit(num_df)
    return pca

def pca_fit_transform(df, n_components, pca=None):
    """Fit & transform a PCA on df
    Args:
        df (DataFrame): pandas dataframe
        n_components (integer||float): integer>indicates the number of components to keep, float>indicates the percentage of variability to keep in data
        pca (pca, optionnal): use pre fitted pca object, indeed skip fit processing
    Returns:
        ndarray: output pca result
    @Author: Nicolas THAIZE
    """
    num_df = get_num_feats(df) 
    if not pca:
        pca = PCA(n_components=n_components)
        pca.fit(num_df)
    return pca.transform(num_df)
    

def pca_plot_var_per_component(pca):
    """plot pca variance explanation by number of components
    Args:
        pca (pca, optional): pca fitted object   
    @Author: Nicolas THAIZE
    """
    plt.plot(np.cumsum(pca.explained_variance_ratio_ * 100))
    plt.xlabel('Nombre de composants')
    plt.ylabel('% de la variance expliquée')
    plt.grid()
    plt.show()

def pca_plot_vars(pca, col_idx_1=2, col_idx_2=1):
    """plot pca components 2 by 2
    Args:
        pca (pca, optional): pca transformed numpy array
        col_idx_1: (integer, optional): pca component index for x axis
        col_idx_2: (integer, optional): pca component index for y axis
    @Author: Nicolas THAIZE
    """
    plt.figure(figsize=(10,7))
    sns.scatterplot(x=pca[:, col_idx_1], y=pca[:, col_idx_2], s=70)
    plt.xlabel('Composant nb %s'%(col_idx_1))
    plt.ylabel('Composant nb %s'%(col_idx_2))
    plt.grid()
    plt.show()

if __name__ == "__main__":
    df = get_data(file_path = "../data/en.openfoodfacts.org.products.csv", nrows=50)
    prep_pca_df = impute_missing(df)
    prep_pca_df = std_scale_df(prep_pca_df)

    result_pca = pca_fit_df(prep_pca_df)
    pca_plot_var_per_component(result_pca)

    pca_transformed = pca_fit_transform(prep_pca_df, 20)
    pca_plot_vars(pca_transformed, 3, 4)
