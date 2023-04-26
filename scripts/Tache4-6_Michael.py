import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import get_data

"""
Task 4.6:

multi_inter_viz(dataset, parameter="heatmap", cols=None):

    This function creates interactive visualizations with columns from the dataframe passed as a parameter.
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

def d_heatmap(dataset, cols):
    # Used to create a Density heatmap plot
    
    if len(cols) != 2 : 
        print("Expect 2 arguments for density heatmap")
    else : 
        fig = px.density_heatmap(dataset, x=cols[0], y=cols[0], text_auto=True)
        fig.show()

def sunburst_graph(dataset, cols, val_col=None):
    # Used to create a Sunburst plot
    
    if val_col is None: 
        print("Expect a val_col parameter for sunburst graph")
    else : 
        fig = px.sunburst(dataset, path=cols, values=val_col)
        fig.show()

def multi_inter_viz(dataset, parameter="heatmap", cols=None, val_col=None):
    # Part 1: selection of columns
    
    if cols is None:
        num_cols = dataset.select_dtypes(include=['int64', 'float64']).columns.tolist()
    else:
        num_cols = dataset[cols].select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Part 2: Check parameters and column length

    if len(num_cols) > 10 and parameter in ["pairplot", "d_heatmap", "sunburst"]: 
        print("More than 10 cols arguments could take several time, please reduce cols number")
        return
    
    # Part 3: Switch parameters to choose graph types

    switcher = {
        "heatmap": lambda data: sns.heatmap(data[num_cols].corr(), cmap='coolwarm'),
        "d_heatmap": lambda data: d_heatmap(dataset, cols),
        "sunburst" : lambda data: sunburst_graph(dataset, cols, val_col),
        }
    
    # Part 4: Run the visualizer with graph parameter

    if parameter in switcher:
        switcher[parameter](dataset[num_cols])
    else:
        print("Invalid parameter - Syntax: multi_var_viz(data, 'parameter', ['col1', 'col2', ...], val_col) \n Parameters: heatmap, d_heatmap, sunburst")


if __name__ == "__main__":
    data = get_data(file_path = "../data/en.openfoodfacts.org.products.csv", nrows=50)
    print(f"data set shape is {data.shape}") 
    multi_inter_viz(data.loc[data.ecoscore_grade !='unknown'], "sunburst", ['ecoscore_grade', 'pnns_groups_1', 'pnns_groups_2'], 'energy-kcal_100g')