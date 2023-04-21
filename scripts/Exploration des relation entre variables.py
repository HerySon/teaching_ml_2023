#######################################
# Exploration relation entre variables#
#######################################
import pandas as pd
import seaborn as sns

def explore_data(df):
    """
    This function takes a Pandas dataframe as input and generates exploratory plots to help understand the relationships between variables.
    """
    # Generate a correlation matrix heatmap to explore relationships between numerical variables
    corr = df.corr()
    sns.heatmap(corr, cmap='coolwarm', annot=True)

    # Generate a countplot to explore relationships between categorical variables
    for col in df.select_dtypes(include=['category']):
        sns.countplot(x=col, data=df)
        
    # Generate a boxplot to explore relationships between a categorical variable and a numerical variable
    for num_col in df.select_dtypes(include=['float', 'int']):
        for cat_col in df.select_dtypes(include=['category']):
            sns.boxplot(x=cat_col, y=num_col, data=df)
