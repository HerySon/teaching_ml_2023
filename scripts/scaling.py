#utilisation des techniques de Scikit-Learn : 
#normaliser le dataset deja nettoye par la methode de  Scikit-Learn  MinMaxScaler.
# definir  min max scaler
from numpy import asarray
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
# charger le dataset 
data= pd.read_csv ()
#En utilisant RobustScaler(), pour  pouvons supprimer les valeurs aberrantes
scaler = RobustScaler()
# standardiser le  jeu de données à l’aide de l’objet scikit-learn StandardScaler.
from sklearn.preprocessing import StandardScaler
# definir  standard scaler
scaler = StandardScaler()
# transformer  data
scaled = scaler.fit_transform(data)
print(scaled)
# a la sortie le scaler cree  une version transformée du jeu de données avec chaque colonne standardisée indépendamment. 
# definir MinMaxScaler
scaler = MinMaxScaler()
# transform er data
scaled = scaler.fit_transform(data)
print(scaled)
# a la sortie le scaler s'adapte à l'ensemble du jeu de données et cree une transdormée  avec chaque colonne normalisée

