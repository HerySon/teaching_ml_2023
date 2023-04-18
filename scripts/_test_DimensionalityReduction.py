import pandas as pd
from LDAReduction import LDAReduction
from MDSReduction import MDSReduction
from IsomapReduction import IsomapReduction
from TSNEReduction import TSNEReduction
from DimensionalityReduction import DimensionalityReduction

# Load your data into a DataFrame
data = pd.read_csv("..\\data\\openfoodfacts.csv", sep="\t", encoding="utf-8", nrows=10)

# Preprocess your data using the preprocess_data function from the main class
dr = DimensionalityReduction(data)
preprocessed_data = dr.preprocess_data(data)

# Visualize the data using t-SNE
tsne = TSNEReduction(preprocessed_data)
reduced_data_tsne = tsne.fit_transform()
dr.plot(reduced_data_tsne, "t-SNE")

# Visualize the data using LDA
lda = LDAReduction(preprocessed_data)
reduced_data_lda = lda.fit_transform()
dr.plot(reduced_data_lda, "LDA")

# Visualize the data using MDS
mds = MDSReduction(preprocessed_data)
reduced_data_mds = mds.fit_transform()
dr.plot(reduced_data_mds, "MDS")

# Visualize the data using Isomap
isomap = IsomapReduction(preprocessed_data)
reduced_data_isomap = isomap.fit_transform()
dr.plot(reduced_data_isomap, "Isomap")


