"""
This code is a sample for testing the SemiSupervisedModel class.

The 'csv' variable is to be modified according to the structure of the project.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from SemiSupervisedModel import SemiSupervisedModel

# Download the Open Food Facts dataset
csv = "..\\data\\openfoodfacts.csv"

# Load the dataset into a Pandas DataFrame
# Because I am just testing my solution, I take only the 10k first rows
df = pd.read_csv(csv, sep="\t", encoding="utf-8", nrows=10000)

# Define the feature columns to use for modeling
"""
These columns are chosen based on the nutritional information provided in the dataset,
after research those are the most used common feature when talking about nutritional value.
"""
feature_cols = [
    "energy_100g",
    "fat_100g",
    "carbohydrates_100g",
    "sugars_100g",
    "fiber_100g",
    "proteins_100g",
    "salt_100g",
    "sodium_100g",
]

# Define the target column for the model
target_col = "nutriscore_grade"

# Clean the dataset by removing rows with missing values in the feature columns or target column.
df.dropna(subset=feature_cols + [target_col], inplace=True)

# Convert the target column to numeric values using LabelEncoder,
"""
Maps the target categories (A, B, C, D, E) to integers (0, 1, 2, 3, 4).
"""
le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col])

# Split the data into labeled and unlabeled subsets for semi-supervised learning.
"""
In this case, we are using a test size of 0.9, which means that 90% of the data will be unlabeled.
"""
labeled_data, unlabeled_data = train_test_split(df, test_size=0.9, random_state=42)

# Initialize the SemiSupervisedModel object
model = SemiSupervisedModel(labeled_data, unlabeled_data, feature_cols, target_col)

# Train the model using LabelPropagation to generate pseudo-labels for the unlabeled data,
# and SVM to fit the combined labeled and pseudo-labeled data.
model.train()

# Evaluate the model by generating a classification report for the predicted labels on the unlabeled data.
model.evaluate()
