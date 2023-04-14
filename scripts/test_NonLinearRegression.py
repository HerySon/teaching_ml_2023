import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from NonLinearReduction import NonLinearReduction

# Read in the data
data = pd.read_csv('..\\data\\openfoodfacts.csv', delimiter='\t', nrows=100)
features = ['energy_100g', 'fat_100g', 'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g', 'salt_100g']
target = 'nutriscore_score'

data.dropna()

# Select the features and target
X = data[features]
y = data[target]

# Perform non-linear dimensionality reduction
n_components = 2
perplexity = 5
random_state = 42
reducer = NonLinearReduction(X, n_components=n_components, perplexity=perplexity, random_state=random_state)
reduced_data = reducer.reduce_dimensions()

# Add the target back to the reduced data for visualization purposes
reduced_data[target] = y

sns.scatterplot(data=reduced_data, x='component_1', y='component_2', hue=target)
plt.show()
