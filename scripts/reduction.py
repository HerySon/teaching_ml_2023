Importer numpy en tant que NP
Importer des pandas en tant que 
Importer uMap. umap_ comme umap
de sklearn. BaseEstimator d’importation de base, TransformerMixin
de sklearn. prétraitement importation StandardScaler
#de l’algorithme UMAP
 def fit(self, X, y=Aucun, precomputed_distances=Aucun):
        si soi-même. métrique == « précalculé »:
            si precomputed_distances est Aucun:
                raise ValueError(
                    « Les distances précalculées doivent être fournies si métrique \
 est précalculé.
                )
            # préparer X à la formation du réseau
            soi-même. _X = X
            # Générer le graphe sur des distances précalculées
            return super(). fit(precomputed_distances, y)
        sinon:
            return super(). fit(X, y)

    def fit_transform(self, X, y=Aucun, precomputed_distances=Aucun):

        si soi-même. métrique == « précalculé »:
            si precomputed_distances est Aucun:
                raise ValueError(
                    « Les distances précalculées doivent être fournies si métrique \
 est précalculé.
        )
            # préparer X à la formation du réseau
            soi-même. _X = X
            # générer le graphique sur des distances précalculées
            return super(). fit_transform(precomputed_distances, y)
        sinon:
            return super(). fit_transform(X, y)
 # Importer la bibliothèque 

de sklearn.  Pipeline d’importation Pipeline


# Créer un pipeline avec UMAPTransformer et un classificateur
 pipeline = pipeline([
    ('umap', UMAPTransformer()),
    ('classifier', RandomForestClassifier())
])

# Diviser l’ensemble de données en train et en ensemble de test
de sklearn. model_selection importer train_test_split
X_train, X_test, y_train, y_test = train_test_split(df. drop(columns=['target']), df['target'], test_size=0.2, random_state=42)

# Former et évaluer le modèle
pipeline. fit(X_train, y_train)
précision = pipeline. score(X_test, y_test)
print(f"Exactitude : {exactitude} »)