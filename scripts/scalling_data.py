import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\Fnac\Desktop\Trello ML\Trello_git/teaching_ml_2023/dataset_clean.csv", encoding='utf-8')

def scale_data(df, method):
  
    """
    Cette fonction prend en entrée un dataframe ensuite elle applique le scaling des données selon la méthode choisie (MinMax ou Standard ou norm ou robust ) et renvoie un daframe.
    paramètres réquis:
        - method: méthode utilisée pour mettre à l'échelle les données ('standard' pour StandardScaler, 'norm' pour Normalizer,'robust' pour RobustScaler, 'minmax' pour MinMaxScaler).
          N.B : Par défaut = 'standard'
        - df : charger à partir du dataset de OpenFoodFact
    """
      
    #Librairies utiles    
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler
    
    # Sélection des variables numériques que nous souhaitons mettre à l'échelle
    colonnes_float = df.select_dtypes(include=['float']).columns
    
    # Gérer les valeurs manquantes en les remplaçant par la médiane
    df[colonnes_float] = df[colonnes_float].fillna(df[colonnes_float].median())

    # Choix du scaler 
    if method == "minmax":
         scaler = MinMaxScaler()
    elif method == "Standard":
        scaler = StandardScaler()
    elif method == "norm":
        scaler = Normalizer()
    elif method == "Robust":
        scaler = RobustScaler()
    else:
        print("Invalid method")
        return None

    # Mettre à l'échelle les variables numériques sélectionnées
    df[colonnes_float] = scaler.fit_transform(df[colonnes_float])

    # Retourner le jeu de données mis à l'échelle
    return df

scale_data(df, "Robust")