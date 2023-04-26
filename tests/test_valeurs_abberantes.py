# Create a sample dataframe with some outliers
data = {'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'B': [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
        'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
df = pd.DataFrame(data)

# Manage outliers using zscore method
df_zscore = manage_outliers(df, method='zscore')
print(df_zscore)

# Manage outliers using IQR method
df_iqr = manage_outliers(df, method='iqr')
print(df_iqr)

# Manage outliers using robust scaling method
df_robust = manage_outliers(df, method='robust')
print(df_robust)
