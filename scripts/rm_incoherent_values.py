def incoherent_values(dataset, check_max_nrj = True, check_sug_carb = True):
    """
    This function aim to clean incoherent values from features. You pass the OpenFoodFact dataset as entry and you get the processed dataset at the end.

    Args :
        dataset -- the dataset who contains OpenFoodFact data (required)
        
        check_max_nrj -- Boolean value to define if remooving the entries with energy for 100g superior to 3700 Kj. 
        (can't be more than this treshold for 100g, the most rich aliments in term of Kj for 100g are oils)
        
        check_sug_carb -- Sugars are members of the carbohydrates family (they are simple carbohydrates), we can't have a value of carbohydrates who is inferior to sugars.
        Boolean value to define if processing remooving of incoherent values (carbohydrates inferior to sugars) or not.

    Returns :
        Dataset of OpenFoodFact cleaned of some incoherent values.
    """
    import numpy as np
    from numpy.testing import assert_equal
    import pandas as pd
    
    assert_equal(type(dataset), type(pd.DataFrame()), err_msg='Input is not Pandas Dataframe.', verbose=True)
    
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
    if check_max_nr == True:
        nutrition_table = nutrition_table.loc[nutrition_table.energy_100g <= 3700]
    if check_sug_carb == True:
        nutrition_table = nutrition_table.loc[nutrition_table.carbohydrates_100g >= nutrition_table.sugars_100g]
    return nutrition_table
