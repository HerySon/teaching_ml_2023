import sklearn
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
# Une valeur aberrante est un élément de données/objet qui s’écarte considérablement du reste des objets (dits normaux).
#  Ils peuvent être causés par des erreurs de mesure ou d’exécution
# pour detecter les valeurs aberrantes (les valeurs plus grandes ou plus petites que les autres ):
 # j'utilisr scores Z ( nombre d’écarts types par rapport à la moyenne)  à travers l'equation z = (x – μ)/σ 
  
  # definir une fonction pour trouver les valeurs aberrantes 
  valeurs aberrantes = [] : definit une lste 
   def detect_outliers( data , seuil  , mean ,std ) :
   data : charger les données
   seuil : definir un seuil  ( toute valeur supérieure au seuil est considerée  comme une  valeur aberrante)
   mean = np.moyenne(data) la moyenne 
    std =  l’écart type des donnée  (std(data))

   # faire une boucle pour  parcourir tous les points de données et calculer le score Z à l'aide de la formule (Xi-moyen) / std.
   pour i in data:
        z_score = (i - mean)/std 
        si (np.abs(z_score) > seuil):
            valeurs aberrantes.append(i)
    renvoyer les valeurs aberrantes 
sample_outliers = detect_outliers(échantillon)
afficher ("Valeurs aberrantes : ", sample_outliers)

 # suppression des  sample_outliers
 pour   i in  sample_outliers:
    a = np.drop (échantillon, np)
afficher 

# remplacer les valeurs manquantes par la valeur de la madiane
médiane = np.médiane(échantillon)
pour i in  sample_outliers:
    c = np.où(échantillon==i, 14, échantillon)
imprimer("Échantillon: ", échantillon)
imprimer("Nouveau tableau: ",c)
# afficher 
  

