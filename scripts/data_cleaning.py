import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import pycountry
import re


def create_test_dataframe():
    """
    This function creates a test pandas DataFrame with 5 columns: product, protein_100g, fat_100g,
    carbohydrates_100g, and serving_size. The DataFrame has 5 rows with different values for each product.

    Parameters:
    None

    Returns:
    pandas.DataFrame: A test DataFrame with 5 rows and 5 columns.
    """
    data = {
        "product": ["A", "B", "C", "D", "E"],
        "protein_100g": [0, -10, 75, 110, 30],
        "fat_100g": [20, 35, 100, -5, 90],
        "carbohydrates_100g": ["30g", "80g", "90g", "5g", "-20g"],
        "serving_size": ["45g (1.5 oz)", "30g (1 oz)", "25g (0.9 oz)", "40g (1.4 oz)", "20g (0.7 oz)"],
        "countries": ["en:FR", "", "en:France", "en:be", "United Kingdom"]
    }
    return pd.DataFrame(data)


def print_columns(dataframe):
    """
    This function prints the entire list of column names in the given dataframe.

    Parameters:
    dataframe (pandas.DataFrame): The pandas DataFrame whose column names you want to print.

    Returns:
    None
    """
    for column in dataframe.columns:
        print(column)


def print_unique_values(column):
    """
    Given a pandas Series (a column from a dataframe), this function prints every unique value.

    Parameters:
    column (pandas.Series): A pandas Series representing a column in a dataframe.

    Returns:
    None
    """
    unique_values = column.unique()
    for value in unique_values:
        print(value)

def remove_na_rows(df, cols=None):
    """
    This function removes rows from the DataFrame where a NaN value is present in any of the specified columns.

    Parameters:
    df (pandas.DataFrame): The input DataFrame from which you want to remove rows containing NaN values.
    cols (list, optional): A list of column names where NaN values should be checked.
                           If None (default), all columns in the DataFrame will be checked for NaN values.

    Returns:
    pandas.DataFrame: A DataFrame with rows containing NaN values in the specified columns removed.
    """
    if cols is None:
        cols = df.columns
    return df[np.logical_not(np.any(df[cols].isnull().values, axis=1))]


def get_country_code(country_name):
    """
    Returns the ISO alpha-2 code of a country given its name.

    Args:
        country_name (str): Name of the country.

    Returns:
        str: ISO alpha-2 code of the country if it is found, else None.
    """
    if not country_name:
        return ''
    try:
        country = pycountry.countries.search_fuzzy(country_name)
        return country[0].alpha_2
    except LookupError:
        return None


def replace_country_name_with_code(df, column_name):
    """
    Replaces country names in a DataFrame column with their corresponding ISO alpha-2 codes.
    Extracts the first part of the string preceding ":" if there is one, and stores it in a new column.

    Args:
        df (pandas.DataFrame): DataFrame containing the column to be modified.
        column_name (str): Name of the column to be modified.

    Returns:
        pandas.DataFrame: Modified DataFrame with the original column replaced by country codes,
        and a new column with the extracted values.
    """
    # create a new column to store the extracted values
    df[f'{column_name}_lang'] = df[column_name].apply(lambda x: x.split(':')[0] if pd.notna(x) and ':' in x else '')

    # apply the get_country_code function to the original column
    df[column_name] = df[column_name].apply(lambda x: x.split(':')[1] if pd.notna(x) and ':' in x else x).apply(
        get_country_code)
    return df


def parse_additives(string_additives):
    """
    This function parses a string containing additives and converts it into a list with the total number of
    unique additives and a sorted list of unique additives.
    It is designed to be used with the pandas .apply() method to apply the transformation to an entire column.

    Parameters:
    string_additives (str): The input string containing additives separated by ']' and '->' delimiters.
                            If the string cannot be parsed, the function will return None.

    Returns:
    list or None: A list containing the total number of unique additives and a sorted list of unique additives,
                  or None if the input string cannot be parsed.
    """
    try:
        additives_set = set()
        for item in string_additives.split(']'):
            token = item.split('->')[0].replace("[", "").strip()
            if token:
                additives_set.add(token)
        return [len(additives_set), sorted(additives_set)]
    except:
        return None


def trans_serving_size(serving_size_str):
    """
    This function extracts the weight value from a serving size string and removes any text.
    It is designed to be used with the pandas .apply() method to apply the transformation to an entire column.

    Parameters:
    serving_size_str (str): The input string containing the serving size weight followed by the unit "g" and
                            possibly other text enclosed in parentheses. If the weight value cannot be extracted,
                            the function will return 0.0.

    Returns:
    float: The serving size weight as a float, or 0.0 if the weight value cannot be extracted from the input string.
    """
    try:
        serving_g = float((serving_size_str.split('(')[0]).replace("g", "").strip())
        return serving_g
    except:
        return 0.0

    """
    make dist. plot on 2x2 grid for up to 4 features
    """

def distplot2x2(food, cols):
    """
    This function creates a distribution plot on a 2x2 grid for up to 4 features.

    Parameters:
    food (pandas.DataFrame): The input DataFrame containing the data to be plotted.
    cols (list): A list of column names (features) in the DataFrame to be plotted.

    Returns:
    None
    """
    sb.set(style="white", palette="muted")
    f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=False)
    b, g, r, p = sb.color_palette("muted", 4)
    colors = [b, g, r, p]
    axis = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
    for n, col in enumerate(cols):
        sb.histplot(food[col].dropna(), color=colors[n], ax=axis[n], kde=True)
    plt.show()

def trans_nutrient_value(value_str):
    """
    This function extracts the nutrient value as a float from a given string.
    If the value cannot be extracted, it returns None.

    Parameters:
    value_str (str): The input string containing the nutrient value followed by the unit "g".

    Returns:
    float or None: The nutrient value as a float, or None if the value cannot be extracted from the input string.
    """
    try:
        value_g = float(value_str.replace("g", "").strip())
        return value_g
    except (ValueError, AttributeError):
        return None

def clean_100g_columns(df):
    """
    This function removes values that are not between 0 and 100 in any columns ending with '_100g' in a given DataFrame.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing columns ending with '_100g' to be cleaned.

    Returns:
    pandas.DataFrame: A DataFrame with values not between 0 and 100 in the specified columns removed.
    """
    columns_100g = [col for col in df.columns if col.endswith('_100g')]

    for col in columns_100g:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(trans_nutrient_value)

        df[col] = df[col].apply(lambda x: x if 0 <= x <= 100 else None)
    return df

