def scale_data(df, method,hyperparams=None):
  
    """
        ------------------------------------------------------------------------------
        Goal : 
            - Applies data scaling according to the chosen method
            (MinMax or Standard or norm or robust)
        ------------------------------------------------------------------------------
        Parameters:
        - df : Dataset to be analyzed
        - method: method used to scale the data ('standard' for StandardScaler, 'norm' 
                 for Normalizer,'robust' for RobustScaler, 'minmax' for MinMaxScaler).
                 N.B: Default = 'standard'
        -----------------------------------------------------------------------------
        Return :
        -  Returns a daframe.
        -----------------------------------------------------------------------------
    """
      
    #Useful libraries  
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler
    
    #Selecting the numeric variables we want to scale
    colonnes_float = df.select_dtypes(include=['number']).columns
    
    #Handle missing values by replacing them with the median
    df[colonnes_float] = df[colonnes_float].fillna(df[colonnes_float].median())

    # Choice of scaler
    if method == "minmax":
        scaler = MinMaxScaler(**hyperparams) if hyperparams else MinMaxScaler()
    elif method == "Standard":
        scaler = StandardScaler() if hyperparams else StandardScaler()
    elif method == "norm":
        scaler = Normalizer() if hyperparams else Normalizer()
    elif method == "Robust":
        scaler = RobustScaler() if hyperparams else RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")

    #Scale selected numeric variables
    df[colonnes_float] = scaler.fit_transform(df[colonnes_float])

    # Return scaled dataset
    return df

if __name__ == "__main__":
    
    # Apply MinMax scaling with custom hyperparameters
    hyperparams = {"feature_range": (0, 1)}
    df_scaled = scale_data(df, method="minmax", hyperparams=hyperparams)
    print(df_scaled)