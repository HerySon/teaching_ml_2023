import pandas as pd

# Charger les données nettoyées
df = pd.read_csv('cleaned_open_food_fact.csv')

# voir les colonnes  non numériques
nonnumeric = df.select_dtypes(exclude=[pd.np.number]).columns.tolist()

# Encoder les colonnes 
data_encoded = pd.get_dummies(df, columns=nonnumeric)
# CSV 
data_encoded.to_csv('encoded_open_food_fact.csv', index=False)