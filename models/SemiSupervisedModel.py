"""
This code defines the SemiSupervisedModel class, which uses two supervised learning algorithms
(LabelPropagation and SVM) to improve data modeling.

The __init__ method initializes the input data, feature columns, and target columns, and divides the labeled data into
training and test sets.

The train method uses LabelPropagation to generate pseudo labels for the unlabeled data and then combines the labeled
and unlabeled data.
Then, an SVM classifier is trained on the combined data.

The method evaluates the performance of the semi-supervised model using SVM prediction on the unlabeled data to generate
a classification report.
"""

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.semi_supervised import LabelPropagation
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


class SemiSupervisedModel:
    def __init__(self, labeled_data, unlabeled_data, feature_cols, target_col):
        """
        Initialize the SemiSupervisedModel class.

        Parameters:
        - labeled_data: a Pandas DataFrame representing the labeled data.
        - unlabeled_data: a Pandas DataFrame representing the unlabeled data.
        - feature_cols: a list of strings representing the feature columns in the data.
        - target_col: a string representing the target column in the data.
        """
        self.labeled_data = labeled_data
        self.unlabeled_data = unlabeled_data
        self.feature_cols = feature_cols
        self.target_col = target_col

        # Split the labeled data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.labeled_data[self.feature_cols],
            self.labeled_data[self.target_col],
            test_size=0.2,
            random_state=42
        )

        # Initialize the LabelPropagation algorithm
        self.label_propagation = LabelPropagation(kernel='knn', n_neighbors=7)

        # Initialize the SVM classifier
        self.svm = SVC(kernel='linear', probability=True)

    def train(self):
        """
        Train the semi-supervised model.
        """
        # Use LabelPropagation to generate pseudo-labels for the unlabeled data
        print("Training LabelPropagation...")
        self.label_propagation.fit(
            self.labeled_data[self.feature_cols],
            self.labeled_data[self.target_col]
        )
        print("Generating pseudo-labels for unlabeled data...")
        self.unlabeled_data[self.target_col] = self.label_propagation.predict(self.unlabeled_data[self.feature_cols])

        # Combine the labeled and pseudo-labeled data
        self.data = pd.concat([self.labeled_data, self.unlabeled_data])

        print("Fitting SVM on combined data...")

        # Fit the SVM classifier on the combined data
        self.svm.fit(self.data[self.feature_cols], self.data[self.target_col])
        print("Training complete.")

    def evaluate(self):
        # Get the actual labels of the unlabeled data
        y_test = self.unlabeled_data[self.target_col]

        # Predict the labels of the unlabeled data using the SVM classifier
        y_pred = self.svm.predict(self.unlabeled_data[self.feature_cols])

        # Generate a classification report for the predicted labels
        """
        The classification report is a summary of the precision, recall, and F1 scores for each class,
        as well as the support (number of samples) for each class and the overall accuracy and macro-averaged scores.
        The 'output_dict' parameter is set to True to return the report as a dictionary, which can be converted to a
        Pandas DataFrame and saved to a CSV file.
        """
        report = classification_report(y_test, y_pred, output_dict=True)

        # Convert the classification report dictionary to a Pandas DataFrame and save it to a CSV file
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv('..\\results\\classification_report.txt', sep='\t')

        # Print the classification report DataFrame to the console
        print(report_df)

    """
Used with test_SemiSupervisedModel.py, the report will look like this :
                  precision  recall  f1-score  support
0                   X          X        X        y
1                   X          X        X        y
2                   X          X        X        y
3                   X          X        X        y
4                   X          X        X        y
accuracy            X          x        x        y
macro avg           X          x        x        y
weighted avg        X          x        x        y

    How to read it : 
- The classification report gives you different metrics for each label (here 0 to 4) as well as some summary metrics.
- X is a number from 0 to 1
- Y is the number of samples in each class

The labels are from 0 to 4 because the target_col is 'nutriscore_grade' encoded with LabelEncoder, so the values 
'a','b','c','d','e' have been translated into '0','1','2','3','4'.

The report created is dynamic according to the chosen features and the encoded target.
    """