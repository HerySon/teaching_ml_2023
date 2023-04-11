import pandas as pd
import numpy as np

class datas_filter():
    def __init__(self) -> None:
        self.dtypes_list = ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]
        pass


    def _get_type_by_range(self, minvalue, maxvalue):
        """Permet de récupérer le type de données le plus adapté pour une feature
            minvalue : int
                Valeur minimale de la feature
            maxvalue : int
                Valeur maximale de la feature
            return : str
                Type de données le plus adapté
        """
        # On vérifie que les valeurs en entrée soient du bon type
        assert isinstance(minvalue, np.int64), "minvalue doit être un int64"
        assert isinstance(maxvalue, np.int64), "maxvalue doit être un int64"

        # On parcourt les types de données
        for dtype in self.dtypes_list:
            # On vérifie que la valeur minimale soit supérieure à 0 pour les types de données unsigned
            if minvalue > 0 and "u" not in dtype:
                continue
            # On vérifie que la valeur minimale et maximale soient dans la range du type de données
            if minvalue >= np.iinfo(dtype).min and maxvalue <= np.iinfo(dtype).max:
                # On retourne le type de données
                return dtype
        else:
            # On retourne "notFound" si aucun type de données n'est trouvé
            return "notFound"


    def filter(self, datas: pd.DataFrame, type : str = "number", 
               category_count : int = 100, nan_percent : int = 0, 
               ordinal_features_names : list = ["ecoscore_grade", "nutriscore_grade"]) -> pd.DataFrame:
        """Permet de filtrer les données en entrée
            datas : pd.DataFrame
                Données en entrée
            type : str
                Type de données à filtrer
                Valeurs possibles : 
                    "number" : valeurs numériques, 
                    "ordonal" : catégories ordinales, 
                    "non-ordinales" : catégories non-ordinales,
                    "categorical" : catégories ordinales et non-ordinales, 
            category_count : int
                Nombre de catégories maximum pour une feature
                (Ne fonctionne que si type != "number")
            nan_percent : int
                Pourcentage de valeurs manquantes maximum pour une feature
            ordinal_features_names : list
                Liste du nom des features catégorielles ordinales
            return : pd.DataFrame
                Données filtrées
        """
        # On vérifie que les données en entrée soient du bon type
        assert isinstance(datas, pd.DataFrame), "datas doit être un pd.DataFrame"
        assert isinstance(type, str), "type doit être un str"
        assert isinstance(category_count, int), "category_count doit être un int"
        assert isinstance(nan_percent, int) or nan_percent >= 0 or nan_percent <= 100, "nan_percent doit être un int entre 0 et 100"
        assert isinstance(ordinal_features_names, list) or ordinal_features_names is None, "ordinal_features_names doit être une liste"

        # On vérifie que le type soit valide
        assert type in ["number", "ordinal", "non-ordinal", "categorical"], "type doit être dans ['number', 'ordinal', 'non-ordinal', 'categorical']"

        # On vérifie que le nombre de catégories soit valide
        assert category_count > 0, "category_count doit être supérieur à 0"

        if type == "number":
            # On récupère les features numériques
            features = datas.select_dtypes(include=["integer", "float"]).columns
        elif type == "ordinal":
            if ordinal_features_names is None:        
                # On retourne le dataframe vide
                return pd.DataFrame()
            # On récupère les features catégorielles ordinales
            features = ordinal_features_names
        elif type == "non-ordinal" or type == "categorical":
            # On récupère les features catégorielles non-ordinales
            features = datas.select_dtypes(include=["category"]).columns
            if type == "non-ordinal":
                for feature in features:
                    # On vérifie que la feature ne soit pas ordonnée et qu'elle ne contienne pas trop de catégories
                    if datas[feature].cat.ordered or datas[feature].nunique() > category_count:
                        features = features.drop(feature)
        result_df = datas[features]
        nans_to_select = result_df.isna().sum()[result_df.isna().sum() / result_df.shape[0] * 100 > nan_percent]
        return result_df.drop(list(nans_to_select.index), axis=1)


    def downcast(self, datas: pd.DataFrame, features = None, show_saved_memory : bool = False) -> pd.DataFrame:
        """Permet de réduire la taille mémoire des données
            datas : pd.DataFrame
                Données en entrée
            features : list
                Liste des features à réduire
                Si None, toutes les features seront réduites
            show_saved_memory : bool
                Affiche la taille mémoire des données avant et après réduction
            return : pd.DataFrame
                Données réduites
        """
        # On vérifie que les données en entrée soient du bon type
        assert isinstance(datas, pd.DataFrame), "datas doit être un pd.DataFrame"
        assert isinstance(features, list) or features is None, "features doit être une liste"
        assert isinstance(show_saved_memory, bool), "show_saved_memory doit être un bool"

        if show_saved_memory:
            # On sauvegarde la taille mémoire des données avant filtrage
            before = datas.memory_usage(index=False, deep=True).sum()

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
                # On récupère le type de données le plus adapté
                dtype = self._get_type_by_range(datas[feature].min(), datas[feature].max())
                if dtype == "notFound":
                    continue
                # On réduit la feature
                datas[feature] = datas[feature].astype(dtype)
            elif datas[feature].dtype == "float64":
                datas[feature] = datas[feature].astype("float16")
        if show_saved_memory:
            # On sauvegarde la taille mémoire des données après filtrage
            after = datas.memory_usage(index=False, deep=True).sum()
            print(f"Taille mémoire des données avant filtrage : {before} octets")
            print(f"Taille mémoire des données après filtrage : {after} octets")
            print(f"Taille mémoire des données sauvegardée : {before - after} octets")
        return datas
        

if __name__ == "__main__":
    filter = datas_filter()
    # Exemple downcast
    datas = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0], "c": ["a", "b", "c"], 
                          "d": [1, 2.0, "c"], "e": [1, 2.0, 2], "f": [1, 2000, np.nan],
                          "g": ["a", "b", "b"], "h": ["a", "a", "a"], "i": ["low", "medium", "high"]})
    datas = filter.downcast(datas, show_saved_memory=True)

    # Exemple filter
    print("Mise en place des filtres :")
    print("Colones numériques : ")
    print(list(filter.filter(datas, type="number", nan_percent=34).columns))
    print("Colones ordinales : ")
    print(list(filter.filter(datas, type="ordinal", ordinal_features_names=["i"]).columns))
    print("Colones non-ordinales (max 2): ")
    print(list(filter.filter(datas, type="non-ordinal", category_count=2).columns))
    print("Colones catégorielles : ")
    print(list(filter.filter(datas, type="categorical").columns))