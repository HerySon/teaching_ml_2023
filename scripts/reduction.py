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
            # prepare X for network formation
            soi-même. _X = X
            # Generate the graph over precalculated distances
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
            
            soi-même. _X = X
            
            return super(). fit_transform(precomputed_distances, y)
        sinon:
            return super(). fit_transform(X, y)
 

de sklearn.  Pipeline d’importation Pipeline


## Create a pipeline with UMAPTransformer and a classifier
 pipeline = pipeline([
    ('umap', UMAPTransformer()),
    ('classifier', RandomForestClassifier())
])

# Divide the dataset into train and test set
de sklearn. model_selection importer train_test_split
X_train, X_test, y_train, y_test = train_test_split(df. drop(columns=['target']), df['target'], test_size=0.2, random_state=42)

# Train and evaluate the model
pipeline. fit(X_train, y_train)
précision = pipeline. score(X_test, y_test)
print(f"Exactitude : {exactitude} »)