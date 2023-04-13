import pandas as pd
import plotly.express as px
from datetime import datetime

class datas_visualizer():
    # Constructor
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

    # Override the __setattr__ function to add validation (ex : dv.datas = pd.DataFrame())
    def __setattr__(self, __name: str, __value) -> None:
        """Override the __setattr__ function to add validation
            ex : self.datas = pd.DataFrame()
        """
        # Prevent the user to change the available plots
        if __name == "_available_plots" and hasattr(self, "_available_plots"):
            raise AttributeError("L'attribut _available_plots est en lecture seule")
        # Check if the datas is a dataframe and not empty
        if __name == "datas" and hasattr(self, "datas"):
            if not isinstance(__value, pd.DataFrame) or __value.empty:
                raise TypeError("datas doit être un DataFrame non vide")
        super().__setattr__(__name, __value)

    # function to set datas
    def set_datas(self, datas: pd.DataFrame):
        self.datas = datas


    # Plot datas using plotly
    def plot(self, x : str, y : str = None, title : str = "", x_label : str = "x", y_label : str = "y",
             color : str = None, figsize : tuple = None, plot_type : str = "scatter", additionnal_params : dict = None,
             save : bool = False, save_path : str = None, show : bool = True) -> None:
        """Plot the datas using plotly
        Args:
            x (str): column to use for the x axis
            y (str, optionnel): column to use for the y axis
            title (str, optionnel): title of the graph
            x_label (str, optionnel): label of the x axis
            y_label (str, optionnel): label of the y axis
            colors (str, optionnel): column to use for the colors
            figsize (tuple, optionnel): size of the graph
            plot_type (str, optionnel): type of the graph
            additionnal_params (dict, optionnel): additionnal params for the graph
            save (bool, optionnel): save the graph
            save_path (str, optionnel): path to save the graph
            show (bool, optionnel): show the graph
        """
        # Check if the dataset if empty or not
        if self.datas is None or not isinstance(self.datas, pd.DataFrame) or self.datas.empty:
            raise ValueError("Le dataframe entrée est vide")

        # Check the method arguments
        # Check colomns
        if x not in self.datas.columns:
            raise ValueError(f"La colonne x : '{x}' n'existe pas (colonne disponible : {self.datas.columns})")
        if y is not None and y not in self.datas.columns:
            raise ValueError(f"La colonne y : '{y}' n'existe pas (colonne disponible : {self.datas.columns})")
        if color is not None and color not in self.datas.columns:
            raise ValueError(f"La colonne color : '{color}' n'existe pas (colonne disponible : {self.datas.columns})")
        
        # Check general params
        if plot_type not in self._available_plots.keys():
            raise ValueError(f"Le type de graphique '{plot_type}' n'est pas disponible, les types disponibles sont : {self._available_plots.keys()}")
        if (isinstance(figsize, tuple) and len(figsize) != 2) or (figsize is not None and not isinstance(figsize, list)):
            raise TypeError("figsize doit être de type tuple de taille 2, reçu : " + str(type(figsize)) + ((" de taille " + str(len(figsize)) if isinstance(figsize, tuple) else "")))
        
        # Check types
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
        
        # List of the graph's params
        params = {
            "x": x,
            "title": title
        }

        # Add optional params
        if y is not None:
            params["y"] = y
        if color is not None:
            params["color"] = color

        if additionnal_params is not None:
            params.update(additionnal_params)

        # Generate the graph type with the params
        fig = self._available_plots[plot_type](self.datas, **params)
        # Change the size of the graph
        if figsize is not None:
            fig.update_layout(
                width=figsize[0],
                height=figsize[1]
            )
        # Change label if not empty
        if x_label is not None or x_label != "":
            fig.update_xaxes(title=x_label)
        if y_label is not None or y_label != "":
            fig.update_yaxes(title=y_label)
        # Show the graph
        if show:
            fig.show()
        # Save the graph
        if save:
            if save_path is None:
                save_path = "../results/plot " + datetime.now().strftime("%d-%m-%Y %H-%M-%S") + ".png"
            fig.write_image(save_path)
        pass


    def exploration_plots(self):
        """Generate the exploration plots (used in the notebook)"""
        df = pd.read_csv("./data/en.openfoodfacts.org.products.csv", sep="\t", nrows=100000)
        # fill all the nan values with "unknown"
        df = df.fillna("unknown")
        # Create a new column that contains 1 if the product contains additives and 0 if not
        df["contains_additives"] = df["additives_tags"].map(lambda x: 0 if x == "unknown" else 1)
        self.datas = df
        # Plot the datas
        # Contains_additives vs nutriscore grade
        self.plot(x="nutriscore_grade", plot_type="hist", title="Présence d'additifs en fonction du nutriscore", 
            color="contains_additives", additionnal_params={"barmode": "group"}, save=False, show=True,
            x_label="Nutriscore", y_label="Nombre de produits")
        # Count of additives vs nutriscore grade
        self.plot(x="nutriscore_grade", y="additives_n", plot_type="hist", title="Nombre d'additifs en fonction du nutriscore", 
            color="nutriscore_grade", additionnal_params={"barmode": "group"}, save=False, show=True,
            x_label="Nutriscore", y_label="Nombre de produits")
        # Ecoscore vs nutriscore
        self.plot(x="ecoscore_score", y="nutriscore_score", plot_type="scatter", title="Eco score en fonction du nutriscore", 
              color="nutriscore_grade", save=False, show=True, x_label="Nutriscore", y_label="Ecoscore")



if "__main__" == __name__:
    # Exemple
    # Load the exemple datas
    datas = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/iris.csv")
    # Generate the visualizer
    dv = datas_visualizer(datas)
    # Show the datas
    dv.plot(x="SepalWidth", y="SepalLength", plot_type="hist", color="Name",
            additionnal_params={"barmode": "group"}, save=True, show=True)
    