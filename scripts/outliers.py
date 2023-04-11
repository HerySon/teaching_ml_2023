import pandas as pd

def handle_outliers(df: pd.DataFrame, how: str = 'standard', numeric_cols: list = None) -> pd.DataFrame:
    
# Set default columns for scaling if not provided
    if numeric_cols is None:
        numeric_cols = ['energy_100g', 'fat_100g', 'carbohydrates_100g', 'proteins_100g']
    
    # Replace missing values with the mean of each column
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    if how == 'standard':
        # Calculate quartiles for each numeric column
        Q1 = df[numeric_cols].quantile(0.25)
        Q3 = df[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1

        # Identify observations with a value outside the range Q1-IQR and Q3+IQR
        outliers = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
        return outliers
    elif how == 'winsorize':
        # Winsorize outliers for each numeric column
        for col in numeric_cols:
            q_low = df[col].quantile(0.01)
            q_high = df[col].quantile(0.99)
            df[col] = df[col].apply(lambda x: q_low if x < q_low else q_high if x > q_high else x)
        return df
    else:
        raise ValueError("Invalid 'how' argument. Allowed values are 'standard' and 'winsorize'.")
