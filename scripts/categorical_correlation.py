#Dataframe
import pandas as pd
import numpy as np
#Viz
import matplotlib.pyplot as plt
#Chi2 test
from  scipy.stats import chi2_contingency  
from scipy.stats import chi2


def cross_table (df,index,columns) : 
    """
    Explore variance of the categorial features in a pandas dataframe using pandas.
    Args:
        df : pandas dataframe
        index : str (column names to plot)
        columns : list of str (The list of column names to plot)

    Returns : 
        Cross table
    """
    #Cross tables
    data_crosstab = pd.crosstab(df.index, df.columns.,
                            margins = False)
    return print(data_crosstab)


def chi_2 (df,columns,alpha):
    """
    Explore independance of two categorial features in a pandas dataframe using pandas.
    Args:
        df : pandas dataframe
        columns : list of str (The list of column names to plot)
        alpha : 

    Returns : 
        Result of chi2
    """
    #Select categorial values
    cat_val = df.select_dtypes(include=['object'])
    #Clear values
    observation = cat_val[[columns]].value_counts().dropna()
    #Chi2
    chi_squared_result=chi2_contingency(observation)
    
    return print("Pearson stats of chi2 : ", chi_squared_result[0], "\n"
    "La p-valeur du test est : ", chi_squared_result[1], "\n")
    print("Le quantile de la loi du chi 2 à 1 degré de liberté associé au niveau de confiance",
    alpha,"est",chi2.ppf(1-alpha, 1))
  

  def plot_chi2 (chi2_result):
    """
    Plot variance of two categorial features in a pandas dataframe using Maplotlib.
    Args:
        chi2_result : result of chi2 test
   
    Returns : 
        Plot chi2
    """

    p_values = pd.Series(chi2_result[1])
    p_values.sort_values(ascending = False , inplace = True)
    return p_values.plot.bar()