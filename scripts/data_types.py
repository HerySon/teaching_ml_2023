import pandas as pd
import numpy as np

#Determine the type of data of a column based on the values
def infer_data_type(column):
    if column.dtype in [np.number, np.float64, np.int64]:
        return 'numerical'

    unique_values = set(x.lower() if isinstance(x, str) else x for x in column.unique() if not pd.isna(x))

    ordinal_categories = {
        # Nutri-Score and Eco-Score
        frozenset(['a', 'b', 'c', 'd', 'e']),

        # NOVA group (food processing level)
        frozenset([1, 2, 3, 4]),

        # Salt, sugar, fat, and saturated fat content
        frozenset(['low', 'moderate', 'high']),
    }

    for ordinal_category in ordinal_categories:
        if unique_values.issubset(ordinal_category):
            return 'ordinal categorical'

    return 'non ordinal categorical'

#Create lists of type of columns
def main(df):
    data_type_counts = {
        'numerical': 0,
        'ordinal categorical': 0,
        'non ordinal categorical': 0,
    }

    data_type_columns = {
        'numerical': [],
        'ordinal categorical': [],
        'non ordinal categorical': [],
    }

    for column_name in df.columns:
        data_type = infer_data_type(df[column_name])
        data_type_counts[data_type] += 1
        data_type_columns[data_type].append(column_name)

        # Debug information
        print(
            f"{column_name}: {data_type} - unique values: {set(x.lower() if isinstance(x, str) else x for x in df[column_name].unique())}")

    print("\nData type counts:")
    for data_type, count in data_type_counts.items():
        print(f"{data_type}: {count}")

    print("\nColumns by data type:")
    for data_type, columns in data_type_columns.items():
        print(f"{data_type}: {', '.join(columns)}")





