import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

def visualize_tsne_numerical_columns(dataframe, param_grid):
    """
    Visualize the numerical columns of a pandas DataFrame using t-SNE, optimizing hyperparameters with grid search.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The input DataFrame containing numerical columns to be visualized.

    param_grid : dict
        The dictionary of hyperparameters to be optimized by GridSearchCV.
        Keys are the hyperparameter names, values are lists of values to try.
       Example : param_grid = {
    'tsne__n_components': [2, 3],
    'tsne__learning_rate': [10, 100, 1000],
    'tsne__perplexity': [5, 10, 20]
}

    Returns
    -------
    None
        This function does not return anything, but it plots the t-SNE visualization of the data.
    """
    # Extract the numerical columns from the DataFrame
    X = dataframe.select_dtypes(include=['number'])

    # Define a pipeline with a StandardScaler and a t-SNE object
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('tsne', TSNE())
    ])

    # Instantiate a GridSearchCV object with 5-fold cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)

    # Fit the data with grid search to find the best hyperparameters
    X_tsne = grid_search.fit_transform(X)

    # Get the best hyperparameters and the best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Visualize the results
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
    plt.show()
