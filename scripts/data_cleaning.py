import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:/Users/Fnac\Desktop/Trello ML/teaching_ml_2023/en.openfoodfacts.org.products.csv", nrows = 10000, sep='\t', encoding='utf-8',low_memory=False)

def taux_remplissage_variables(df, tx_threshold=50):
    
    
    """
    Calcule le taux de nullité de chaque variable dans un DataFrame et retourne une liste des variables avec un taux de nullité supérieur ou égal à un seuil donné.
    
    Paramètres :
    - df : DataFrame, l'ensemble de données à analyser
    - tx_threshold : seuil de nullité au-dessus duquel une variable est considérée comme ayant un taux de nullité élevé (par défaut 50%)
    
    Retourne :
    - high_null_rate : DataFrame, une liste des variables avec leur taux de nullité supérieur ou égal au seuil spécifié, triée par ordre décroissant de taux de nullité
    """
    
    # Calcul du taux de nullité pour chaque variable
    null_rate = ((df.isnull().sum() / df.shape[0])*100).sort_values(ascending=False).reset_index()
    null_rate.columns = ['Variable','Taux_de_Null']
    
    # Sélection des variables avec un taux de nullité supérieur ou égal au seuil spécifié
    high_null_rate = null_rate[null_rate.Taux_de_Null >= tx_threshold]
    
    return high_null_rate

def fonction_taux_remplissage_features(df):
    
    """
    Calcule le taux de remplissage de chaque variable dans un DataFrame et affiche un graphique à barres horizontales qui montre le taux de remplissage pour chaque variable.
    
    Paramètres :
    - df : DataFrame, l'ensemble de données à analyser
    """
    
    # Calcul du taux de nullité pour chaque variable en utilisant la fonction taux_remplissage_variables
    filling_features = taux_remplissage_variables(df, 0)
    
    # Calcul du taux de remplissage en soustrayant le taux de nullité de 100%
    filling_features["Taux_de_Null"] = 100-filling_features["Taux_de_Null"]
    
    # Tri des résultats par ordre décroissant de taux de remplissage
    filling_features = filling_features.sort_values("Taux_de_Null", ascending=False)
    
    # Création du graphique à barres horizontales avec Seaborn
    fig = plt.figure(figsize=(20, 35))
    font_title = {'family': 'serif',
                'color':  '#114b98',
                'weight': 'bold',
                'size': 18,
                }
    sns.barplot(x="Taux_de_Null", y="Variable", data=filling_features, palette="flare")
    plt.axvline(linewidth=2, color = 'r')
    plt.title("Taux de remplissage des variables dans le jeu de données (%)", fontdict=font_title)
    plt.xlabel("Taux de remplissage (%)")
    
    return plt.show()


def nettoyer_donnees(df):
    
    """
    Nettoyage du dataset
    
    Paramètres :
    - df : DataFrame, l'ensemble de données à analyser
    """
    fonction_taux_remplissage_features(df)
    
    colonnes_a_garder = []

    #Colonne selectionné après analyse de toute les colonnes
    colonnes_a_garder = ['cities', 'code', 'created_datetime','states','product_name','countries_en',"categories","states","pnns_groups_2","ingredients_text","additives_n","nutriscore_grade","brands"]

    #Sélection des colonnes ayant un suffixe '_100g' et ajout de ces colonnes aux colonnes déjà sélectionnées 
    for column in df.columns:
        if '_100g' in column: colonnes_a_garder.append(column)

    # Suppression de toutes les colonnes sauf celles à garder
    colonnes_a_supprimer = [col for col in df.columns if col not in colonnes_a_garder]
    df_garder = df.drop(colonnes_a_supprimer, axis=1)

    # A travers le plot effectué pour le taux de remplissage nous remarquons qu'il y a des features qui ont plus de 30 à 40 % de valeurs manquantes
    # Calcul du pourcentage de valeurs manquantes dans chaque colonne
    pourcentages_val_manquantes = df_garder.isnull().mean() * 100

    # Sélection des colonnes qui ont plus de 40% de valeurs manquantes
    colonnes_val_manquantes = pourcentages_val_manquantes[pourcentages_val_manquantes > 40].index

    # Suppression de toutes les colonnes ayant plus de 40% de valeurs manquantes
    df_garder=df_garder.drop(colonnes_val_manquantes,axis=1)

    #Suppression des doublons dans tout le dataframe
    df_garder.drop_duplicates(inplace=True)

    # Liste des colonnes ayant pour type float
    colonnes_float = df_garder.select_dtypes(include=['float']).columns

    # Suppression les valeurs incohérentes
    for col in colonnes_float:
        # Remplacer les valeurs négatives par des NaN
        df_garder.loc[df_garder[col] < 0, col] = np.nan

    # Enregistrer les modifications dans un nouveau fichier CSV
    df_garder.to_csv('dataset_clean.csv', index=False)

    return df_garder

nettoyer_donnees(df)