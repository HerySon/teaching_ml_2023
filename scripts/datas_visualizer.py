from typing import Union
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime

class datas_visualizer():
    def __init__(self, datas : pd.DataFrame = None) -> None:
        """Data visualizer
        Args:
            datas (dataframe): input dataframe
        """
        self.datas = datas
        self._available_plots = {
            "line": px.line,
            "bar": px.bar,
            "hist": px.histogram,
            "box": px.box,
            "violin": px.violin,
            "scatter": px.scatter,
            "scatter_3d": px.scatter_3d,
            "scatter_matrix": px.scatter_matrix,
        }
        pass

    def __setattr__(self, __name: str, __value) -> None:
        """Fonction de validation des attributs à l'assignation
            ex : self.datas = pd.DataFrame()
        """
        # Empêche la modification de l'attribut _available_plots
        if __name == "_available_plots" and hasattr(self, "_available_plots"):
            raise AttributeError("L'attribut _available_plots est en lecture seule")
        if __name == "datas" and hasattr(self, "datas"):
            if not isinstance(__value, pd.DataFrame) or __value.empty:
                raise TypeError("datas doit être un DataFrame non vide")
        super().__setattr__(__name, __value)


    def set_datas(self, datas: pd.DataFrame):
        self.datas = datas


    # Plot les données à l'aide de pyplot
    def plot(self, x : str, y : str = None, title : str = "", x_label : str = "x", y_label : str = "y",
             color : str = None, figsize : tuple = None, plot_type : str = "scatter", additionnal_params : dict = None,
             save : bool = False, save_path : str = None, show : bool = True) -> None:
        """Plot les données
        Args:
            x (str): colonne à utiliser pour l'axe des abscisses
            y (str, optionnel): colonne à utiliser pour l'axe des ordonnées
            title (str, optionnel): titre du graphique
            x_label (str, optionnel): label de l'axe des abscisses
            y_label (str, optionnel): label de l'axe des ordonnées
            colors (str, optionnel): colonne résponsable de la couleur du graphique
            figsize (tuple, optionnel): taille du graphique
            plot_type (str, optionnel): type de graphique
            additionnal_params (dict, optionnel): paramètres additionnels
            save (bool, optionnel): sauvegarde le graphique
            save_path (str, optionnel): chemin de sauvegarde
            show (bool, optionnel): affiche le graphique
        """
        if self.datas is None or not isinstance(self.datas, pd.DataFrame) or self.datas.empty:
            raise ValueError("Le dataframe entrée est vide")

        # Vérifications des arguments de la fonction
        # Vérification des colones
        if x not in self.datas.columns:
            raise ValueError(f"La colonne x : '{x}' n'existe pas (colonne disponible : {self.datas.columns})")
        if y is not None and y not in self.datas.columns:
            raise ValueError(f"La colonne y : '{y}' n'existe pas (colonne disponible : {self.datas.columns})")
        if color is not None and color not in self.datas.columns:
            raise ValueError(f"La colonne color : '{color}' n'existe pas (colonne disponible : {self.datas.columns})")
        
        # Vérification des paramètres généraux
        if plot_type not in self._available_plots.keys():
            raise ValueError(f"Le type de graphique '{plot_type}' n'est pas disponible, les types disponibles sont : {self._available_plots.keys()}")
        if (isinstance(figsize, tuple) and len(figsize) != 2) or (figsize is not None and not isinstance(figsize, list)):
            raise TypeError("figsize doit être de type tuple de taille 2, reçu : " + str(type(figsize)) + ((" de taille " + str(len(figsize)) if isinstance(figsize, tuple) else "")))
        
        # Vérifications des types
        if not isinstance(save, bool):
            raise TypeError("save doit être de type bool, reçu : " + str(type(save)))
        if not isinstance(show, bool):
            raise TypeError("show doit être de type bool, reçu : " + str(type(show)))
        if not isinstance(title, str):
            raise TypeError("title doit être de type str, reçu : " + str(type(title)))
        if not isinstance(x_label, str):
            raise TypeError("x_label doit être de type str, reçu : " + str(type(x_label)))
        if not isinstance(y_label, str):
            raise TypeError("y_label doit être de type str, reçu : " + str(type(y_label)))
        if save_path is not None and not isinstance(save_path, str):
            raise TypeError("save_path doit être de type str, reçu : " + str(type(save_path)))
        if additionnal_params is not None and not isinstance(additionnal_params, dict):
            raise TypeError("additionnal_params doit être de type dict, reçu : " + str(type(additionnal_params)))
        
        # Liste des paramètres du graphique
        params = {
            "x": x,
            "title": title,
            "labels": {
                "x": x_label,
                "y": y_label
            }
        }

        # Ajout des paramètres optionnels
        if y is not None:
            params["y"] = y
        if color is not None:
            params["color"] = color

        if additionnal_params is not None:
            params.update(additionnal_params)

        # Création du graphique en utilisant plotly
        fig = self._available_plots[plot_type](self.datas, **params)
        # Modification de la taille du graphique
        if figsize is not None:
            fig.update_layout(
                width=figsize[0],
                height=figsize[1]
            )
        # Affichage du graphique
        if show:
            fig.show()
        # Sauvegarde du graphique
        if save:
            if save_path is None:
                save_path = "../results/plot " + datetime.now().strftime("%d-%m-%Y %H-%M-%S") + ".png"
            fig.write_image(save_path)
        pass
        

if "__main__" == __name__:
    # Exemple
    # Chargement des données
    datas = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/iris.csv")
    # Création de l'objet
    dv = datas_visualizer(datas)
    # Affichage des données
    dv.plot(x="SepalWidth", y="SepalLength", plot_type="hist", color="Name",
            additionnal_params={"barmode": "group"}, save=True, show=True)
    