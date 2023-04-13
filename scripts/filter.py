def Filter_df(df, num_cols=None, ordinal_cols=None, nominal_cols=None, cat_threshold=50):
    """
    df : data
    +num_cols: List of column names for numeric variables
     +ordinal_cols: List of column names for ordinal categorical variables
    +nominal_cols: List of column names for nominal categorical variables
    +cat_threshold: Maximum number of categories for nominal variables
    """
     # Convert numeric columns with low cardinality to int or float for memory optimization
    if num_cols:
        for col in num_cols:
            if np.issubdtype(df[col].dtype, np.integer):
                if df[col].nunique() < len(df) * 0.5:
                    df[col] = pd.to_numeric(df[col], downcast='integer')
                else:
                    df[col] = pd.to_numeric(df[col], downcast='signed')
            else:
                if df[col].nunique() < len(df) * 0.5:
                    df[col] = pd.to_numeric(df[col], downcast='float')
                else:
                    df[col] = pd.to_numeric(df[col], downcast='signed')

    # Select numeric columns
    if num_cols:
        num_df = df[num_cols]
    else:
        num_df = pd.DataFrame()

    # Select ordinal categorical columns
    if ordinal_cols:
        ordinal_df = df[ordinal_cols].astype('category')
    else:
        ordinal_df = pd.DataFrame()

    # Select nominal categorical columns with low cardinality
    if nominal_cols:
        nominal_df = pd.DataFrame()
        for col in nominal_cols:
            if df[col].nunique() <= cat_threshold:
                nominal_df[col] = df[col].astype('category')
    else:
        nominal_df = pd.DataFrame()

    # Combine selected columns into one DataFrame
    selected_df = pd.concat([num_df, ordinal_df, nominal_df], axis=1)

    return selected_df