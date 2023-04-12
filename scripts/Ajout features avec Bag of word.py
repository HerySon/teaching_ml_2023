##########################################################
# Creation de features additionelles Bag of words        #
##########################################################
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def create_bag_of_words_features(df, text_column, max_features=5000, stop_words=None):
    """
    This function takes in a pandas DataFrame and a text column name and creates additional features
    using Bag of Words technique.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the text data.
    text_column (str): The name of the column containing the text data.
    max_features (int): The maximum number of features to use in the CountVectorizer. Default is 5000.
    stop_words (str or list): The stop words to use in the CountVectorizer. Default is None.
    
    Returns:
    pandas.DataFrame: A new DataFrame with additional features created using Bag of Words technique.
    @Auteur: Massamba GUEYE
    """
    # Define the CountVectorizer object with custom parameters
    count_vectorizer = CountVectorizer(max_features=max_features, stop_words=stop_words)

    # Apply common preprocessing steps to the text data
    df[text_column] = df[text_column].str.lower()  # Convert text to lowercase
    df[text_column] = df[text_column].str.replace('[^\w\s]','')  # Remove punctuation
    df[text_column] = df[text_column].str.replace('\d+','')  # Remove digits

    # Fit and transform the preprocessed text data to create a bag of words representation
    bag_of_words = count_vectorizer.fit_transform(df[text_column])

    # Create a DataFrame of the bag of words representation
    bag_of_words_df = pd.DataFrame(bag_of_words.toarray(), columns=count_vectorizer.get_feature_names())

    # Concatenate the bag of words DataFrame with the original DataFrame
    df_with_features = pd.concat([df, bag_of_words_df], axis=1)

    return df_with_features
#Test