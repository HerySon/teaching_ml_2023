import matplotlib.pyplot as plt
import multidict as multidict
from categorical_viz import explode_col_by_delimiter
from data_loader import get_data

def remove_prefixes(serie, prefixes=['en:']):
    """Remove prefix in serie
    Args:
        serie (serie): serie to use
        prefixes (array string, optional): array of prefixes to remove from serie entries
            By default ['en:']  
    Returns:
        serie: output serie
    """
    for prefix in prefixes:
        serie = serie.str.removeprefix(prefix)
    return serie

def lower_text(text):
    """Lower the text
    Args:
        text (string): text to use
    Returns:
        String: output text
    """
    if isinstance(text, str):
        return text.lower()
    return text

def get_country_number_of_rows(df, country_col_name='countries'):
    """Count number of countries appearance in df countries column
    Args:
        df (DataFrame): dataframe to use
        country_col_name ( string, optional): country column name in dataframe
            By default countries  
    Returns:
        Serie: countries count serie idx: country name, value: count
    """
    countries = explode_col_by_delimiter(df, country_col_name, new_col_name="country")
    countries = remove_prefixes(countries['country'])
    countries = countries.apply(lower_text).value_counts()
    return countries

def plot_countries(serie, min_y_value=50, max_y_value=1e10):
    """Bar plot countries count
    Args:
        serie (Serie): Serie to use
        min_y_value (int, optional): min x value to plot
            By 50
        max_y_value (int, optional): min x value to plot
            By 1E10
    Returns:
        Serie: countries count serie idx: country name, value: count
    """
    countries_zoom = serie[serie.between(min_y_value, max_y_value)]
    fig, ax = plt.subplots()
    plt.bar(countries_zoom.index, countries_zoom)
    plt.xticks(rotation = 45)
    plt.title("Number of rows by country")
    plt.xlabel("Name of the country")
    plt.ylabel("Number of rows")
    fig.subplots_adjust(bottom=0.3)
    plt.show()
    
if __name__ == "__main__":
    data = get_data(file_path = "../data/en.openfoodfacts.org.products.csv", nrows=10000)
    print(f"data set shape is {data.shape}")

    countries = get_country_number_of_rows(data)
    plot_countries(countries)   