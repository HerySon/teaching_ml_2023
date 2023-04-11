###########################################################
# Nettoyage du jeu de données des valeurs problématiques  #
###########################################################
# Importation des bibliotéques
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import seaborn as sns
#Importationde 17000 premiere lignes de la base de données, soit 10%
datas = pd.read_csv(r"D:\Bibliothèques\Documents\M1 data Ynov\Maj 3 Machine Learning\OpenFoodFact\data\en.openfoodfacts.org.products.csv", nrows = 1700, sep='\t', encoding='utf-8')
# Infos sur la dataset
datas.info()
#Fonction pour afficher les nombres de valeurs null
def null_factor(df, tx_threshold=50):
    # df: notre dataframe de départ
    # tx_threshold : Le seuil de nullité qui est fixé par défaut à 50% et au dessus duquel la vraibale sera supprimée
  null_rate = ((datas.isnull().sum() / datas.shape[0])*100).sort_values(ascending=False).reset_index()
  null_rate.columns = ['Variable','Taux_de_Null']
  high_null_rate = null_rate[null_rate.Taux_de_Null >= tx_threshold]
  return high_null_rate
#test pour savoir les variables avec 100% de valeurs null
full_null_rate = null_factor(datas, 100)
full_null_rate
#Nous allons regarder le taux de remplissage des variables graphiquement et fixer un seuil de suppression à 25% de taux de remplissage
filling_features = null_factor(datas, 0)
filling_features["Taux_de_Null"] = 100-filling_features["Taux_de_Null"]
filling_features = filling_features.sort_values("Taux_de_Null", ascending=False) 

#Seuil de suppression
sup_threshold = 25

fig = plt.figure(figsize=(20, 35))

font_title = {'family': 'serif',
              'color':  '#114b98',
              'weight': 'bold',
              'size': 18,
             }

sns.barplot(x="Taux_de_Null", y="Variable", data=filling_features, palette="flare")
#Seuil pour suppression des varaibles
plt.axvline(x=sup_threshold, linewidth=2, color = 'r')
plt.text(sup_threshold+2, 65, 'Seuil de suppression des variables', fontsize = 16, color = 'r')

plt.title("Taux de remplissage des variables dans le jeu de données (%)", fontdict=font_title)
plt.xlabel("Taux de remplissage (%)")
plt.show()
#Liste des variables à conserver
features_to_conserve = list(filling_features.loc[filling_features['Taux_de_Null']>=sup_threshold, 'Variable'].values)
#Liste des variables supprimées
deleted_features = list(filling_features.loc[filling_features['Taux_de_Null']<sup_threshold, 'Variable'].values)

#Nouveau Dataset avec les variables conservées
datas = datas[features_to_conserve].sort_values(["created_datetime","last_modified_datetime"], ascending=True)
datas.sample(5)
#Fonction pour conserver les varibales suffixées avec _100g 
def search_componant(df, suffix='_100g'):
    #df: dataframe de départ
    # Suffixe de la variable
  componant = []
  for col in df.columns:
      if '_100g' in col: componant.append(col)
  df_subset_columns = df[componant]
  return df_subset_columns
#Affichage
df_subset_nutients = search_componant(datas,'_100g')
df_subset_nutients.head()
datas = datas[df_subset_nutients.notnull().any(axis=1)]
datas.shape
# Suppression des doublons en fonction du code
datas.drop_duplicates(subset ="code", keep = 'last', inplace=True)
datas[(datas["product_name"].isnull()==False) 
      & (datas["brands"].isnull()==False)].groupby(by=["product_name","brands"])["code"].nunique().sort_values(ascending=False)
# Suppression des doublons sur marque et produit en conservant les valeurs nulles
datas = datas[(~datas.duplicated(["product_name","brands"],keep="last")) 
      | ((datas['product_name'].isnull()) & (datas['brands'].isnull()))]
datas.shape
#Suppresion des variables redondantes par exemple le cas des variables suffixées par _tags ou _en qui ne font que reprendre d'autres features traduites ou simplifiées.
#Exemple :
category_columns = ['categories','categories_tags','categories_en']
datas[datas[category_columns].notnull().any(axis=1)][['product_name'] + category_columns].sample(5)
def search_redundant_col(df):
  redundant_columns = []
  for col in df.columns:
    if "_en" in col:
      en = col.replace('_en','')
      tags = col.replace('_en','_tags')
      print("{:<20} 'Sans suffixe' -> {} ; 'Suffixe _tags' -> {}".format(col,
                                                                        en in df.columns, tags in df.columns))
      if en in df.columns : 
        redundant_columns.append(en)
      if tags in df.columns : 
        redundant_columns.append(tags)
  
    if '_tags' in col:
      tags_2 = col.replace('_tags','')
      print("{:<20} 'Suffixe _tags' -> {} ;".format(tags_2, tags_2 in df.columns))
      if tags_2 in df.columns :
        redundant_columns.append(col)

  return redundant_columns
#Suppression
datas.drop(search_redundant_col(datas), axis=1, inplace=True)
#Les dates également comportent une certaine redondance. Entre les timestamp et les dates au format "yyyy-mm-dd", il est nécessaire d'en éliminer :
datas['created_datetime'] = pd.to_datetime(datas['created_t'], unit='s')
datas['last_modified_datetime'] = pd.to_datetime(datas['last_modified_t'], unit='s')
datas = datas.drop(['created_t','last_modified_t'], axis=1)
datas.head()
#Suppression tous les produits qui n'ont ni nom, ni catégorie 
datas_cleaned = datas[~((datas.product_name.isnull()) 
                        & ((datas.pnns_groups_1 == "unknown") 
                           | (datas.main_category_en == "unknown")))]
#On supprime les lignes dont toutes les numerical_features sont à 0 ou nulles
datas_cleaned = datas_cleaned.loc[~((datas_cleaned[numerical_features]==0) | (datas_cleaned[numerical_features].isnull())).all(axis=1)]
#On supprime les lignes contenant des valeurs négatives et des max aberrants
datas_cleaned = datas_cleaned[~(datas_cleaned[numerical_features] < 0).any(axis=1)]
datas_cleaned = datas_cleaned[~(datas_cleaned[numerical_features].isin([999999,9999999])).any(axis=1)]
# supprimer les lignes dont au moins 1 des variables de nutriments est supérieur au seuil pour les variabes _100g
g_per_100g_features = ['proteins_100g','fat_100g','carbohydrates_100g','sugars_100g','salt_100g',
                       'sodium_100g','saturated-fat_100g','fiber_100g']
datas_cleaned = datas_cleaned[~(datas_cleaned[g_per_100g_features] > 100).any(axis=1)]
#saturated-fat_100g < fat_100g, de même sodium_100g < salt_100g.On supprime les lignes qui ne remplissement pas es conditions
datas_cleaned = datas_cleaned[~((datas_cleaned['saturated-fat_100g'] > datas_cleaned['fat_100g']) 
                                | (datas_cleaned['sodium_100g'] > datas_cleaned['salt_100g']))]
#Nous allons donc supprimer toutes les lignes dont la variable energy_100g est supérieur à 3700 (ou 900 kcal/100g).
datas_cleaned = datas_cleaned[~((datas_cleaned['energy_100g'] > 3700) 
                                | (datas_cleaned['energy-kcal_100g'] > 900))]
#la médiane et l'écart-type pour éliminer les outliers
#On initialise l'écart-type et la médiane
sigma_features = ['additives_n','serving_quantity']
sigma = [0 for _ in range(len(sigma_features))]
median = [0 for _ in range(len(sigma_features))]
#Puis on complètes les valeurs avec le dataset sans les valeurs nulles
for i in range(len(sigma_features)):
  median[i] = datas_cleaned[pd.notnull(datas_cleaned[sigma_features[i]])][sigma_features[i]].median()
  serie = datas_cleaned[pd.notnull(datas_cleaned[sigma_features[i]])][sigma_features[i]]
  serie = serie.sort_values()
  sigma[i] = np.std(serie[:-25])
#
for i in range(len(sigma_features)):
    col = sigma_features[i]
    threshold = (median[i] + 5*sigma[i])
    print('{:30}: suppression de la ligne si valeur > {}'.format(col, round(threshold,3)))
    mask = datas_cleaned[col] > threshold
    datas_cleaned = datas_cleaned.drop(datas_cleaned[mask].index)