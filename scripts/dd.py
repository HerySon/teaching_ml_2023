
import pandas as pd
import numpy as np

class datas_filter():
    def __init__(self) -> None:
        pass

    def filter(self, datas: pd.DataFrame, type : str = "numerical", category_count : int = 100, nan_percent : int = 0) -> pd.DataFrame:
        """Permet de filtrer les données en entrée
            datas : pd.DataFrame
                Données en entrée
            type : str
                Type de données à filtrer
                Valeurs possibles : 
                    "numerical" : valeurs numériques, 
                    "ordonal" : catégories ordinales, 
                    "non-ordinales" : catégories non-ordinales,
                    "categorical" : catégories ordinales et non-ordinales, 
            category_count : int
                Nombre de catégories maximum pour une feature 
                (Ne fonctionne que si type != "numerical")
            nan_percent : int
                Pourcentage de valeurs manquantes maximum pour une feature
            return : pd.DataFrame
                Données filtrées
        """
        # On vérifie que les données en entrée soient du bon type
        assert isinstance(datas, pd.DataFrame), "datas doit être un pd.DataFrame"
        assert isinstance(type, str), "type doit être un str"
        assert isinstance(category_count, int), "category_count doit être un int"

        # On vérifie que le type soit valide
        assert type in ["numerical", "ordinal", "non-ordinal", "categorical"], "type doit être dans ['numerical', 'ordinal', 'non-ordinal', 'categorical']"

        # On vérifie que le nombre de catégories soit valide
        assert category_count > 0, "category_count doit être supérieur à 0"

        if type == "numerical":
            # On récupère les features numériques
            features = datas.select_dtypes(include=["integer", "float"]).columns
        elif type == "ordinal":
            # On récupère les features catégorielles ordinales
            features = datas.select_dtypes(include=["category"]).columns
            for feature in features:
                # On vérifie que la feature soit ordonnée et qu'elle ne contienne pas trop de catégories
                if not datas[feature].cat.ordered or datas[feature].nunique() > category_count:
                    features = features.drop(feature)
        elif type == "non-ordinal":
            # On récupère les features catégorielles non-ordinales
            features = datas.select_dtypes(include=["category"]).columns
            for feature in features:
                # On vérifie que la feature ne soit pas ordonnée et qu'elle ne contienne pas trop de catégories
                if datas[feature].cat.ordered or datas[feature].nunique() > category_count:
                    features = features.drop(feature)
        elif type == "categorical":
            # On récupère les features catégorielles
            features = datas.select_dtypes(include=["category"]).columns
        result_df = datas[features]
        nans_to_select = result_df.isna().sum()[result_df.isna().sum() / result_df.shape[0] * 100 > nan_percent]
        return result_df.drop(list(nans_to_select.index), axis=1)


    def downcast(self, datas: pd.DataFrame, features = None) -> pd.DataFrame:
        """Permet de réduire la taille mémoire des données
            datas : pd.DataFrame
                Données en entrée
            features : list
                Liste des features à réduire
            return : pd.DataFrame
                Données réduites
        """
        # On vérifie que les données en entrée soient du bon type
        assert isinstance(datas, pd.DataFrame), "datas doit être un pd.DataFrame"
        assert isinstance(features, list) or features is None, "features doit être une liste"

        # Si aucune feature n'est spécifiée, on prend toutes les features
        if features is None:
            features = datas.columns

        # On parcourt les features
        for feature in features:
            if datas[feature].dtype == "object":
                # On verifie que la feature ne soit pas multitype
                if datas[feature].apply(type).nunique() > 1:
                    continue
                # Si la feature n'est que du type string, on la réduit
                datas[feature] = datas[feature].astype("category")
                continue
            elif datas[feature].dtype == "int64":
                datas[feature] = pd.to_numeric(datas[feature], downcast="integer")
                continue
            elif datas[feature].dtype == "float64":
                datas[feature] = pd.to_numeric(datas[feature], downcast="float")
                continue
        return datas
        

if __name__ == "__main__":
    filter = datas_filter()
    # Exemple downcast
    datas = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0], "c": ["a", "b", "c"], 
                          "d": [1, 2.0, "c"], "e": [1, 2.0, 2], "f": [1, 2000, np.nan],
                          "g": ["a", "b", "b"], "h": ["a", "a", "a"], "i": ["low", "medium", "high"]})
    print("Avant downcast :")
    print(datas.dtypes)
    datas = filter.downcast(datas)
    print("Après downcast :")
    print(datas.dtypes)

    # Exemple filter
    print("Mise en place des filtres :")
    print("Colones numériques : ")
    print(list(filter.filter(datas, type="numerical", nan_percent=34).columns))
    print("Colones ordinales : ")
    print(list(filter.filter(datas, type="ordinal").columns))
    print("Colones non-ordinales (max 2): ")
    print(list(filter.filter(datas, type="non-ordinal", category_count=2).columns))
    print("Colones catégorielles : ")
    print(list(filter.filter(datas, type="categorical").columns))

