import pandas as pd
#Based on the dataset legend, try to attribute a type to each column
def count_fields_by_category(df):
    fields = df.columns
    counts = {
        'dates': 0,
        'iso8601_dates': 0,
        'tags': 0,
        'lang_tags': 0,
        'nutriment_100g': 0,
        'nutriment_serving': 0,
        'other': 0
    }

    for field in fields:
        if field.endswith('_t'):
            counts['dates'] += 1
        elif field.endswith('_datetime'):
            counts['iso8601_dates'] += 1
        elif field.endswith('_tags'):
            counts['tags'] += 1
        elif len(field) == 2 and field.isalpha():
            counts['lang_tags'] += 1
        elif field.endswith('_100g'):
            counts['nutriment_100g'] += 1
        elif field.endswith('_serving'):
            counts['nutriment_serving'] += 1
        else:
            counts['other'] += 1

    return counts





