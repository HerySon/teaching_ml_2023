def scale_data(dataset, how=standard):
    """
    This function aim to standardize values in dataset. You pass the OpenFoodFact dataset as entry and you get the standardized dataset at the end.

    Args :
        dataset -- the dataset who contains OpenFoodFact data (required)
        how -- method used to scale data. 'norm' for Normalizer, 'standard' for StandardScaler, 'minmax' for MinMaxScaler. Default = standard

    Returns :
        Dataset of OpenFoodFact with standardized values.
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
    if how == standard:
        scaler = StandardScaler()
    elif how == norm:
        scaler = Normalizer()
    elif how == minmax:
        scaler = MinMaxScaler()
    vars_to_scale = dataset.select_dtypes(include=['number']).columns
    scaler.fit(dataset[vars_to_scale])
    dataset[vars_to_scale] = scaler.transform(X[vars_to_scale])
    return dataset
