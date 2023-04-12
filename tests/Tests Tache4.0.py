####################################################
#test visualisation des données univariés avec Iris#
####################################################
import pandas as pd
import seaborn as sns
import univariate_visualizations as uv

# Load sample data
df = sns.load_dataset("iris")

# Create a wordcloud of the unique values in the 'species' column
uv.create_wordcloud(df, "species", stopwords=["setosa", "versicolor", "virginica"])

# Create a density plot of the 'petal_length' column
uv.create_density_plot(df, "petal_length", xlim=(0, 10), title="Density Plot of Petal Length")

# Count the unique values in the 'species' column and create a barplot
uv.create_value_counts(df, "species", title="Barplot of Species")

# Count the unique values in the 'species' column and create a histogram
uv.create_value_counts(df, "species", title="Histogram of Species", barplot=False)
