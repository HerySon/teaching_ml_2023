import pandas as pd

class datas_filter():
    def __init__(self) -> None:
        pass

    def filter(self, datas: pd.DataFrame, type = "numerical") -> pd.DataFrame:
        """Permet de filtrer les données en entrée
            datas : pd.DataFrame
                Données en entrée
            type : str
                Type de données à filtrer
                Valeurs possibles : "numerical", "ordonal", "non-ordinales"
            return : pd.DataFrame
                Données filtrées
        """
        pass

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
    datas_filter = datas_filter()
    # Test downcast
    datas = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0], "c": ["a", "b", "c"], 
                          "d": [1, 2.0, "c"], "e": [1, 2.0, 2], "f": [1, 2000, 3]})
    print("Avant downcast :")
    print(datas.dtypes)
    datas = datas_filter.downcast(datas)
    print("Après downcast :")
    print(datas.dtypes)