import pandas as pd

data = pd.read_csv("openfoodfacts.csv", sep="\t", encoding="utf-8")

# les colonnes non utils
data.drop(columns=["url", "creator", "created_t", "last_modified_t", "pnns_groups_1", "pnns_groups_2"], inplace=True)

# les lignes des valeurs manquantes et supprimer
data.dropna(inplace=True)

# Convertir les colonnes 
data["code"] = data["code"].astype(str)
data["energy_100g"] = data["energy_100g"].astype(float)
data["fat_100g"] = data["fat_100g"].astype(float)
data["carbohydrates_100g"] = data["carbohydrates_100g"].astype(float)
data["proteins_100g"] = data["proteins_100g"].astype(float)
# Filtrer les produits avec des valeurs nutritionnelles invalides ou irréalistes
data = data[(data["energy_100g"] > 0) & (data["fat_100g"] >= 0) & (data["carbohydrates_100g"] >= 0) & (data["proteins_100g"] >= 0)]

# # Supprimer les doublons en fonction du code produit
data.drop_duplicates(subset=["code"], keep="first", inplace=True)

