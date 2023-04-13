import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

#Create a test dataframe
def create_test_dataframe():
    data = {
        "product": ["A", "B", "C", "D", "E"],
        "protein_100g": [0, -10, 75, 110, 30],
        "fat_100g": [20, 35, 100, -5, 90],
        "carbohydrates_100g": ["30g", "80g", "90g", "5g", "-20g"],
        "serving_size": ["45g (1.5 oz)", "30g (1 oz)", "25g (0.9 oz)", "40g (1.4 oz)", "20g (0.7 oz)"],
    }
    return pd.DataFrame(data)

#Print every column not just a snipet
def print_columns(dataframe):
    for column in dataframe.columns:
        print(column)

#Remove every rows where a NaN is present in one of the columns input
def remove_na_rows(df, cols=None):
    if cols is None:
        cols = df.columns
    return df[np.logical_not(np.any(df[cols].isnull().values, axis=1))]

#Replace country name to it's 2 char code (run it with .apply(trans_country_name) for it to change an entire column)
def trans_country_name(string):
    try:
        country_name = string.split(',')[0]
        if country_name in dictCountryName2Code:
            return dictCountryName2Code[country_name]
    except:
        return None

#Parse additives column values into a list (run it with .apply(parse_additives) for it to change an entire column)
def parse_additives(string_additives):
    try:
        additives_set = set()
        for item in string_additives.split(']'):
            token = item.split('->')[0].replace("[", "").strip()
            if token:
                additives_set.add(token)
        return [len(additives_set), sorted(additives_set)]
    except:
        return None


#Get the weight value while removing the text (run it with .apply(trans_serving_size) for it to change an entire column)
def trans_serving_size(serving_size_str):
    try:
        serving_g = float((serving_size_str.split('(')[0]).replace("g", "").strip())
        return serving_g
    except:
        return 0.0

    """
    make dist. plot on 2x2 grid for up to 4 features
    """
#Create a plot from 2x2 grid for 1-4 features, cols need to be a list
def distplot2x2(food, cols):
    sb.set(style="white", palette="muted")
    f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=False)
    b, g, r, p = sb.color_palette("muted", 4)
    colors = [b, g, r, p]
    axis = [axes[0,0],axes[0,1],axes[1,0],axes[1,1]]
    for n, col in enumerate(cols):
        sb.histplot(food[col].dropna(), color=colors[n], ax=axis[n], kde=True)
    plt.show()




#Similar to trans_serving_size but instead of replacing errors with 0 it sets them as None
def trans_nutrient_value(value_str):
    try:
        value_g = float(value_str.replace("g", "").strip())
        return value_g
    except (ValueError, AttributeError):
        return None

#Given a dataframe, remove values that are not between 0 and 100 in any columns ending with _100g
def clean_100g_columns(df):
    columns_100g = [col for col in df.columns if col.endswith('_100g')]

    for col in columns_100g:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(trans_nutrient_value)

        df[col] = df[col].apply(lambda x: x if 0 <= x <= 100 else None)
    return df


