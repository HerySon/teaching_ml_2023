import pandas as pd
from sklearn.decomposition import KernelPCA

def reduce_dim_NLPCA(df, kernel='rbf', gamma=1.0, n_components=2):
    """
    df : data
    """
    #      Selecting numeric columns
    numeric_cols = df.select_dtypes(include=[int, float]).columns
    
    # Removing columns with missing values
    df_numeric = df[numeric_cols].dropna()
    
    #Création a KernelPCA object for nonlinear data transformation
    kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma)
    
    #Fit and transform from the KernelPCA object to the df_numeric data
    X_kpca = kpca.fit_transform(df_numeric)
    
     #Création of a pandas dataframe with projected data in n_components dimensions
    df_proj = pd.DataFrame(X_kpca, columns=[f'dim{i}' for i in range(1, n_components+1)])
    
   # Adding product information
    df_info = df[['code', 'product_name', 'categories', 'brands']]
    df_proj = pd.concat([df_proj, df_info], axis=1)
    
    return df_proj

    """
df = pd.read_csv('openfoodfacts.csv', delimiter='\t', low_memory=False)

df_nlpca = reduce_dim_NLPCA(df, kernel='cosine', gamma=0.1, n_components=3)
display
print(df_nlpca.head())
    """