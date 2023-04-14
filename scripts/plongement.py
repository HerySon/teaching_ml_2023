def preprocessdata(data):
    """
        Preprocess the data with StandardScalar and Label Encoder
    :param data: input dataframe of training or test set 
    """
    labels = data['LABELS']
    features = data.drop(['LABELS'], axis=1)
    columns = features.columns
    enc = LabelEncoder()
    enc.fit(labels)
    labels = enc.transform(labels)
    features = StandardScaler().fit_transform(features)
    return features, labels, columns, data['LABELS']

Xtrain, Ytrain, COLUMNS, ACTUAL_LABELS = preprocessdata(TRAINDATA)
Xtest, Ytest, _, _ = preprocessdata(TESTDATA)
"""
Before applying feature extraction techniques such as PCA or LDA, we will check the performance of the Random Forest model on the input dataset.

"""
def applyrandomforest(trainX, testX, trainY, testY):
    """
        Apply Random forest on input dataset.
    """
    start = time.process_time()
    forest = RandomForestClassifier(n_estimators=700, max_features='sqrt', max_depth=15)
    forest.fit(trainX, trainY)
    print("Time Elapsed: %s secs" % (time.process_time() - start))
    prediction = forest.predict(testX)
    print("Classification Report after applying Random Forest: ")
    print("----------------------------------------------------")
    print(classification_report(testY, prediction))
    """
    Let's apply the PCA to our input dataset
    """
# Fitting the PCA algorithm with our Data
pca = PCA()
pca.fit(trainX)
# Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')  # for each component
plt.title('Segmentation Dataset Explained Variance')
plt.show(block=True)

"""
 Apply t-SNE for better visualization
"""
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(trainX, trainY)
# transform the data onto the first two principal components
trainX_lda = lda.transform(trainX)
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
    trainX_tsne = tsne.fit_transform(trainX)
    plt.figure(figsize=(10, 10))
    plt.xlim(trainX_tsne[:, 0].min(), trainX_tsne[:, 0].max())
    plt.ylim(trainX_tsne[:, 1].min(), trainX_tsne[:, 1].max())
    for i in range(len(trainX_tsne)):
        # actually plot the digits as text instead of using scatter
        plt.text(trainX_tsne[i, 0], trainX_tsne[i, 1], str(trainY[i]),
                 color=colors[trainY[i]], fontdict={'weight': 'bold', 'size': 9})
    plt.xlabel("t-SNE feature 0")
    plt.ylabel("t-SNE feature 1")
    plt.show(block=True)