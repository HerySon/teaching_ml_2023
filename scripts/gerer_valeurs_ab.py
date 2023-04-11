import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\Fnac\Desktop\Trello ML\Trello_git/teaching_ml_2023/dataset_clean.csv", encoding='utf-8')


def gestion_valeurs_aberrantes(df, colonnes=None, seuil=3, action="delete"):
    
    """
        ------------------------------------------------------------------------------------------------------------------------------------
        Objectif : 
        ------------
        Cette fonction prend en entrée un dataframe, une liste de colonnes, un seuil et une action à effectuer en cas de valeurs aberrantes.
        Elle renvoie un dataframe modifié selon l'action choisie.
        --------------
        Paramètres
        --------------
        df: dataframe contenant les données
        colonnes: liste des noms des colonnes à traiter (par défaut, toutes les colonnes sont traitées)
        seuil: seuil utilisé pour déterminer les valeurs aberrantes (par défaut, seuil = 3)
        action: action à effectuer en cas de valeurs aberrantes ("delete" pour supprimer, "mean" pour remplacer par la moyenne)
        ---------------
        En sortie: dataframe modifié selon l'action choisie
        -------------------------------------------------------------------------------------------------------------------------------------
        
    """
    # Si aucune colonne n'est spécifiée, on traite toutes les colonnes
    if colonnes is None:
        colonnes = df.columns
    
    # Calcule de la moyenne et de l'écart-type pour chaque colonne spécifiée
    moyennes = np.mean(df[colonnes], axis=0)
    ecarts_types = np.std(df[colonnes], axis=0)
    
    # Détermination des valeurs aberrantes pour chaque colonne spécifiée
    valeurs_aberrantes = (df[colonnes] < moyennes - seuil * ecarts_types) | (df[colonnes] > moyennes + seuil * ecarts_types)
    
    # Suppression ou remplacement des lignes contenant des valeurs aberrantes
    if action == "delete":
        df_modifie = df[~np.any(valeurs_aberrantes, axis=1)]
    elif action == "mean":
        df_modifie = df.copy()
        for c in colonnes:
            df_modifie[c] = np.where(valeurs_aberrantes[c], moyennes[c], df_modifie[c])
    else:
        raise ValueError("L'action choisie n'est pas valide. Veuillez choisir 'delete' ou 'mean'.")
    
    return df_modifie

# Sélection des variables numériques dont nous souhaitons gérer les valeurs abbérantes
colonnes_float = df.select_dtypes(include=['float']).columns
df_sans_aberrantes = gestion_valeurs_aberrantes(df,colonnes_float,seuil=3, action="delete")
df_sans_aberrantes