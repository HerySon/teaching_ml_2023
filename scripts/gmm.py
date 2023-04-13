from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

def gmm_clustering(dataset, n_components_range=(2,10)):
    """
    This function will fit a Gaussian Mixture Model. You pass the dataset as entry and you get the fitted model at the end.
    Args:
        dataset -- the dataset to cluster (required).
        n_components_range -- a tuple (start, end) for the range of the number of components in the mixture model.
                              Default is (2,10).
    Returns:
        Fitted Gaussian Mixture Model.
    """
    # Verifying if dataset input is a pd.DataFrame().
    assert isinstance(dataset, pd.DataFrame), 'Input is not Pandas Dataframe.'

    # Removing rows with missing values
    dataset = dataset.dropna()

    # Verifying if there are at least 2 rows after removing missing values
    assert dataset.shape[0] >= 2, 'The dataset must contain at least 2 rows after removing missing values.'

    # Verifying if all columns are numeric
    assert all(dataset.dtypes.apply(lambda x: np.issubdtype(x, np.number))), 'All columns must be numeric.'

    # Scaling the dataset
    scaler = StandardScaler()
    dataset_scaled = scaler.fit_transform(dataset)

    # Finding the best number of components using the Bayesian Information Criterion (BIC)
    bic = []
    for n_components in range(n_components_range[0], n_components_range[1]+1):
        gmm = GaussianMixture(n_components=n_components, random_state=42).fit(dataset_scaled)
        bic.append(gmm.bic(dataset_scaled))

    # Plotting the BIC as a function of the number of components
    plt.plot(range(n_components_range[0], n_components_range[1]+1), bic, marker='o')
    plt.xlabel("Number of components")
    plt.ylabel("Bayesian Information Criterion")
    plt.title("BIC as a function of number of components")

    # Choosing the best number of components
    best_n_components = np.argmin(bic) + n_components_range[0]

    # Fitting the final model with the best number of components
    gmm = GaussianMixture(n_components=best_n_components, random_state=42).fit(dataset_scaled)

    return gmm
