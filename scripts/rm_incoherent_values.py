def incoherent_values(dataset):
    import pandas as pd
    """
    This function aim to clean incoherent values from features. You pass the OpenFoodFact dataset as entry and you get the processed dataset at the end.

    Args :
        dataset -- the dataset who contains OpenFoodFact data (required)

    Returns :
        Dataset of OpenFoodFact cleaned of some incoherent values.
    """
    nutrition_table_cols = ["energy_100g",
                            "fat_100g",
                            "carbohydrates_100g",
                            "sugars_100g",
                            "proteins_100g",
                            "salt_100g"]
    nutrition_table = dataset[nutrition_table_cols]
    for col in nutrition_table.columns:
        if col not in ["energy_100g"]:
            nutrition_table = nutrition_table.loc[nutrition_table[col] <= 100]
        nutrition_table = nutrition_table.loc[nutrition_table[col] >= 0]
    nutrition_table = nutrition_table.loc[nutrition_table.energy_100g <= 3700]
    nutrition_table = nutrition_table.loc[nutrition_table.carbohydrates_100g >= nutrition_table.sugars_100g]
    dataset = dataset.loc[nutrition_table.index, :]
    return dataset
