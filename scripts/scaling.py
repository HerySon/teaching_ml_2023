def scale_data(dataset, how='standard', nrm = 'l2', qtl_rng = (25.0, 75.0)):
    """
    This function aim to standardize values in dataset. You pass the OpenFoodFact dataset as entry and you get the standardized dataset at the end.

    Args :
        dataset -- the dataset who contains OpenFoodFact data (required)
        
        how -- method used to scale data. 'norm' for Normalizer, 'standard' for StandardScaler, 'minmax' for MinMaxScaler, 'robust' for RobustScaler. 
        Default = 'standard'
        
        nrm -- The norm to use to normalize each non zero sample. If norm=’max’ is used, values will be rescaled by the maximum of the absolute values.
        Possibles values are : ['l1', 'l2', 'max']. Default = 'l2'
        
        qtl_rng -- tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0. 
        Quantile range used to calculate the interquartile range for each feature in the training set. Default = (25.0, 75.0)

    Returns :
        Dataset of OpenFoodFact with standardized values.
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler
    if how == 'standard':
        scaler = StandardScaler()
    elif how == 'norm':
        scaler = Normalizer(norm=nrm)
    elif how == 'minmax':
        scaler = MinMaxScaler()
    elif how == 'robust':
        scaler = RobustScaler(quantile_range=qtl_rng)
    vars_to_scale = dataset.select_dtypes(include=['number']).columns
    scaler.fit(dataset[vars_to_scale])
    dataset[vars_to_scale] = scaler.transform(X[vars_to_scale])
    return dataset
