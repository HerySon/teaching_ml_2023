from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
def apply_random_forest(train_X, test_X, train_y, test_y, n_estimators=100, max_depth=None):
    # Initialize the Random Forest model with the given hyperparameters
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    # Fit the model to the training data
    model.fit(train_X, train_y)

    # Use the model to make predictions on the test data
    y_pred = model.predict(test_X)

    # Calculate the accuracy of the model on the test data
    accuracy = accuracy_score(test_y, y_pred)

    return model, y_pred, accuracy
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(train_X, train_y)
# transform the data onto the first two principal components
trainX_lda = lda.transform(train_X)
colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525",
          "#A83683", "#4E655E"]
plt.figure(figsize=(10, 10))
plt.xlim(trainX_lda[:, 0].min(), trainX_lda[:, 0].max())
plt.ylim(trainX_lda[:, 1].min(), trainX_lda[:, 1].max())
for i in range(len(trainX_lda)):
    # actually plot the digits as text instead of using scatter
    plt.text(trainX_lda[i, 0], trainX_lda[i, 1], str(trainY[i]),
             color=colors[trainY[i]], fontdict={'weight': 'bold', 'size': 9})
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.show(block=True)
"""""
We will apply t-SNE which is available at scikit-learn.s
"""""
 # Apply tSNE from Manifold learning for better visualization
    tsne = TSNE(random_state=21)
    # use fit_transform instead of fit, as TSNE has no transform method
    trainX_tsne = tsne.fit_transform(train_X)
    plt.figure(figsize=(10, 10))
    plt.xlim(trainX_tsne[:, 0].min(), trainX_tsne[:, 0].max())
    plt.ylim(trainX_tsne[:, 1].min(), trainX_tsne[:, 1].max())
    for i in range(len(trainX_tsne)):
        # actually plot the digits as text instead of using scatter
        plt.text(trainX_tsne[i, 0], trainX_tsne[i, 1], str(train_y[i]),
                 color=colors[train_y[i]], fontdict={'weight': 'bold', 'size': 9})
    plt.xlabel("t-SNE feature 0")
    plt.ylabel("t-SNE feature 1")
    plt.show(block=True)