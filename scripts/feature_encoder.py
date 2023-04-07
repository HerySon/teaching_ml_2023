def encoding_vars(dataset, big_memory = False, cols_to_split_but_big_memory = ["categories", "ingredients_tags"],
                  cols_to_split=["nutrient_levels_tags", "food_groups_tags", "allergens", "ingredients_analysis_tags", "labels_fr"], 
                 ordinal_cols = ["ecoscore_grade", "nutriscore_grade", "nova_group"],
                 dummies_cols = ["brands_tags", "origins_fr", "generic_name", "product_name", "first_packaging_code_geo",
                    "additives_fr", "pnns_groups_1", "pnns_groups_2", "main_category"]):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import OrdinalEncoder
    """
    This function encode non-numeric features. You pass the OpenFoodFact dataset as entry and you get the encoded dataset at the end.
    Will be different encoding depending on some factors. Features with ordinal order are encoded with OrdinalEncoder. 
    Features without ',' are encoded with OneHotEncoder. Features with ',' are manually encoded, like a OneHotEncoder would have done, but before we split the datas.

    Args :
        dataset -- the dataset who contains OpenFoodFact data (required)
        big_memory -- boolean argument to define if you will run the function with a machine with really big memory or not (True or False)
        cols_to_split -- list of columns containing values separated with ',' who represent a list of features. Basically, we will split the values and each value will become a new feature.
        cols_to_split_but_big_memory -- list to columns who will be processed like cols_to_split, but only if big_memory value is True.
        ordinal_cols -- list of columns to process with OrdinalEncoder. (relation of order between values of the column)
        dummies_cols -- list of columns to process with OneHotEncoder. (due to increasing dimensions resulting in OneHotEncoding, we only OneHotEncode features with a limited number of distinct values)  

    Returns :
        Encoded dataset of OpenFoodFact.
    """
    if big_memory == True:
        cols_to_split = cols_to_split + cols_to_split_but_big_memory
    # Ordinals (OrdinalEncoder)
    for i in ordinal_cols:
        dataset[i].fillna("unknown", inplace=True)
    oe = OrdinalEncoder(
        categories=[["not-applicable", "unknown", "e", "d", "c", "b", "a"], ["unknown", "e", "d", "c", "b", "a"],
                    ["unknown", 4.0, 3.0, 2.0, 1.0]])
    oe.fit(dataset[ordinal_cols])
    dataset[ordinal_cols] = oe_ecoscore_grade.transform(dataset[ordinal_cols])
    dataset.drop(ordinal_cols, axis=1, inplace=True)
    # Dummies (OneHotEncoding)
    dataset = dataset.join(pd.get_dummies(dataset[dummies_cols]))
    dataset.drop(dummies_cols, axis=1, inplace=True)
    # Splits
    for i in cols_to_split:
        splitted = dataset[i].str.split(",", expand=True)
        columnar = pd.concat([splitted[0], splitted[1]])
        i = 2
        while i <= int(splitted.columns[-1]):
            columnar = pd.concat([columnar, splitted[i]])
            i += 1
        list_of_cat = columnar.value_counts().index
        dataset[i] = dataset[i].astype(str)
        for j in list_of_cat:
            dataset[j] = np.nan
            for u in range(len(dataset)):
                if dataset.loc[u, i].__contains__(j):
                    dataset.at[u, j] = 1
                else:
                    dataset.at[u, j] = 0
    dataset.drop(
        ["creator", "abbreviated_product_name", "brands", "categories_tags", "origins_tags", "origins", "categories_fr",
         "last_modified_by", "traces_tags", "packaging", "packaging_tags", "packaging_fr", "packaging_test",
         "manufacturing_places", "manufacturing_places_tags", "labels", "labels_tags", "emb_codes", "emb_codes_tags",
         "cities", "cities_tags", "purchase_places", "stores", "countries", "countries_tags", "countries_fr",
         "ingredients_text", "allergens_fr", "traces", "traces_fr", "serving_size", "serving_quantity",
         "no_nutrition_data", "additives", "additives_tags", "food_groups", "food_groups_fr", "states", "states_tags",
         "states_fr", "brand_owner", "owner", "popularity_tags", "main_category_fr"], axis=1, inplace=True)
    return dataset
