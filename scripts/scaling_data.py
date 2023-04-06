import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def scaling_data(dataset, how='standard'):
    """
    Cette fonction prend en entrée un dataset au format CSV et applique le scaling
    des données selon la méthode choisie (MinMax ou Standard).
    La fonction renvoie un DataFrame avec les données scalées.
    """
    df = pd.read_csv(dataset, delimiter='\t', low_memory=False)
    
    # Sélectionner les colonnes pour le scaling
    numeric_cols = ['energy_100g', 'fat_100g', 'carbohydrates_100g', 'proteins_100g']
    
    # Remplacer les valeurs manquantes par la médiane de chaque colonne
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Créer le scaler correspondant
    if how == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    
    # Appliquer le scaling aux colonnes numériques
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Renvoyer le DataFrame avec les données scalées
    return df
