import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import IncrementalPCA

def pca_chunk(filepath, chunksize, threshold):
    """
    Applies Incremental Principal Component Analysis (IPCA) to a dataset in chunks of size `chunksize`,
    calculates the number of components that explain a given threshold of the total variance, and
    returns the dataset with the chosen number of components along with the number of components chosen.
    
    Parameters:
        filepath (str): The path to the dataset that has to be cleaned and composed of numerical columns to be processed.
        chunksize (int): The number of rows to process at a time.
        threshold (float): The cumulative variance ratio threshold to use.
    
    Returns:
        tuple: A tuple containing the processed dataset with the chosen number of components and
        the number of components chosen to explain the given threshold of the total variance.
    """
    
    # Initialize IPCA with a batch size of 10
    ipca = IncrementalPCA(batch_size=50)
    
    # Read the dataset in chunks of size `chunksize`
    f = pd.read_csv(filepath, chunksize=chunksize, iterator=True)
    
    # Fit IPCA incrementally on each chunk of the dataset
    for chunk in f:
        # Fill missing values with 0 and fit the chunk to IPCA
        ipca.partial_fit(chunk.fillna(0))
    
    # Calculate the cumulative explained variance ratio for each principal component
    cum_expl_var_ratio = np.cumsum(ipca.explained_variance_ratio_)
    
    # Calculate the cumulative variance explained by each principal component
    cum_var = np.cumsum(ipca.explained_variance_ratio_)
    
    # Find the number of principal components needed to explain the given threshold of the total variance
    k = np.argmax(cum_var > threshold)
    
    # Print the number of principal components needed to explain the given threshold of the total variance
    print(f"Number of components explaining {threshold:.0%} variance: {k}")
    
    # Plot the cumulative explained variance as a function of the number of principal components
    sns.lineplot(x=range(1, len(cum_var)+1), y=cum_var)
    plt.title("Cumulative Explained Variance explained by the components")
    plt.ylabel("Cumulative Explained Variance")
    plt.xlabel("Principal components")
    plt.axvline(x=k, color="k", linestyle="--")
    plt.axhline(y=threshold, color="r", linestyle="--")
    plt.show()
    
    # Read the dataset again in chunks of size `chunksize` and transform it using the chosen number of components
    f = pd.read_csv(filepath, chunksize=chunksize, iterator=True)
    # Get the first chunk of the dataset
    transformed = ipca.transform(f.get_chunk(chunksize).fillna(0))
    # Transform the remaining chunks of the dataset
    for chunk in f:
        transformed = np.vstack([transformed, ipca.transform(chunk.fillna(6))])
    
    # Return the processed dataset with the chosen number of components and the number of components chosen
    return transformed[:,:k+1], k+1