import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import get_data

"""
Task 4.1:

multi_var_viz(dataset, parameter="heatmap", cols=None):

    This function creates visualizations with columns from the dataframe passed as a parameter.
    The arguments passed as parameters are:
    - dataset:
        which represents the dataset
    - parameter:
        to choose graph types
    - cols:
        contains all columns to use

    Algorithmic specificity:
        - Part 1: Selection of int and float columns
        - Part 2: Check parameters and column length
        - Part 3: Switch parameters to choose graph types
        - Part 4: Run the visualizer with graph parameter

    Function return:
        - The function returns nothing but prints a graph on the terminal
"""

def third_dim_plot_graph(dataset, cols):
    # Used to create a 3D scatter plot
    
    if len(cols) != 4 : 
        print("Expect 4 arguments for 3D scatter")
    else : 
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(dataset[cols[1]], dataset[cols[2]], dataset[cols[3]], c=dataset[cols[0]])
        ax.set_xlabel(cols[1])
        ax.set_ylabel(cols[2])
        ax.set_zlabel(cols[3])
        plt.show()

def multi_var_viz(dataset, parameter="heatmap", cols=None):
    # Part 1: selection of columns
    
    if cols is None:
        num_cols = dataset.select_dtypes(include=['int64', 'float64']).columns.tolist()
    else:
        num_cols = dataset[cols].select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Part 2: Check parameters and column length

    if len(num_cols) > 10 and parameter in ["pairplot", "3d_scatter", "kde_pairplot"]: 
        print("More than 10 cols arguments could take several time, please reduce cols number")
        return
    
    # Part 3: Switch parameters to choose graph types

    switcher = {
        "heatmap": lambda data: sns.heatmap(data[num_cols].corr(), cmap='coolwarm'),
        "pairplot": lambda data: sns.pairplot(data, vars=data[num_cols].columns.tolist()),
        "3d_scatter" : lambda data : third_dim_plot_graph(data[num_cols], num_cols),
        "kde_pairplot" : lambda data : sns.pairplot(data, diag_kind='kde')
        }
    
    # Part 4: Run the visualizer with graph parameter

    if parameter in switcher:
        switcher[parameter](dataset[num_cols])
    else:
        print("Invalid parameter - Syntax: multi_var_viz(data, 'parameter', ['col1', 'col2', ...] \n Parameters: heatmap, pairplot, 3d_scatter, kde_pairplot")

if __name__ == "__main__":
    data = get_data(file_path = "../data/en.openfoodfacts.org.products.csv", nrows=50)
    print(f"data set shape is {data.shape}") 
    multi_var_viz(data, "kde_pairplot", ['energy_100g','fat_100g','carbohydrates_100g','proteins_100g'])
