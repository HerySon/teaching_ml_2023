#######################################
# Exploration relation entre variables#
#######################################
import pandas as pd
import seaborn as sns

def explore_data(df):
    """
    This function takes a Pandas dataframe as input and generates exploratory 
     plots to help understand the relationships between variables.
    """
    # Generate a correlation matrix heatmap to explore relationships between numerical variables
    corr = df.corr()
    sns.heatmap(corr, cmap='coolwarm', annot=True)
