#0.0_Filtrage_GuillaumeCHUPE

def filter_dataframe(df, num_cols=None, ordinal_cols=None, nominal_cols=None, downcast_cols=None, max_categories=None):
    """
    Filter and select relevant columns in the input data frame, distinguishing between numeric, ordinal, and nominal columns.

    Args:
        df (DataFrame): The input dataframe to filter.
        num_cols (list of str): List of column names for numeric columns.
        ordinal_cols (list of str): List of column names for ordinal columns.
        nominal_cols (list of str): List of column names for nominal columns.
        downcast_cols (list of str): List of column names for columns that need to be downcasted.
        max_categories (int): Maximum number of categories allowed for nominal columns. If None, all nominal columns are kept.

    Returns:
        DataFrame: The filtered dataframe.
    """
    # initialize empty lists for each column type
    numeric_cols = []
    ordinal_cols = [] if ordinal_cols else []
    nominal_cols = [] if nominal_cols else []
    downcast_cols = [] if downcast_cols else []

    # iterate through all columns
    for col in df.columns:
        if df[col].dtype in [np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]:
            # numeric column
            numeric_cols.append(col)
        elif ordinal_cols and col in ordinal_cols:
            # ordinal column
            ordinal_cols.append(col)
        elif nominal_cols and col in nominal_cols:
            # nominal column
            if max_categories is None or df[col].nunique() <= max_categories:
                nominal_cols.append(col)
        if downcast_cols and col in downcast_cols:
            # downcast column
            df[col] = pd.to_numeric(df[col], downcast='integer')

    # return filtered dataframe with selected columns
    selected_cols = numeric_cols + ordinal_cols + nominal_cols
    return df[selected_cols]
