def filter_columns(df, numeric=True, ordinal=True, nominal=True, max_categories=None, downcast=False):
    """
    -------------------------------------------------------------------------
    Filters and selects relevant columns in a dataframe.
    ------------------------------------------------------------------------
    Parameters:
    ------------------------------------------------------------------------
    df : The input dataframe to filter columns for.
    numeric : bool, optional (default=True)
        Whether to include numerical columns in the output.
    ordinal : bool, optional (default=True)
        Whether to include ordinal categorical columns in the output.
    nominal : bool, optional (default=True)
        Whether to include nominal categorical columns in the output.
    max_categories : int, optional (default=None)
        The maximum number of categories a categorical variable can have to be included in the output.
        If None, all categorical variables will be included.
    downcast : bool, optional (default=False)
        Whether to downcast numerical variables to a smaller datatype to save memory.
    ---------------------------------------------------------------------------
    Returns:
    ---------------------------------------------------------------------------
        The filtered dataframe with relevant columns selected.
    """

    # Separating numerical, ordinal categorical and nominal categorical columns
    numeric_cols = list(df.select_dtypes(include=['number']).columns)
    ordinal_cols = []
    nominal_cols = []
    for col in df.select_dtypes(include=['category']):
        if len(df[col].unique()) == 2:
            ordinal_cols.append(col)
        elif max_categories is None or len(df[col].unique()) <= max_categories:
            nominal_cols.append(col)

    # Selecting relevant columns based on variable types
    relevant_cols = []
    if numeric:
        relevant_cols += numeric_cols
    if ordinal:
        relevant_cols += ordinal_cols
    if nominal:
        relevant_cols += nominal_cols
    
    # Downcasting numerical columns if specified
    if downcast:
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], downcast='signed')

    # Returning filtered dataframe
    return df[relevant_cols]

if __name__ == "__main__":
    
    filtered_df = filter_columns(df, nominal=True, ordinal=True, numeric=True, max_categories=3, downcast=True)

    print(filtered_df.dtypes)