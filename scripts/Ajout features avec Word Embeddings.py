##########################################################
# Creation de features additionelles avec word embedding #
##########################################################
import gensim
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

# Define a function to tokenize the text data
def tokenize(text):
    return word_tokenize(text.lower())

# Define a function to create Word2Vec embeddings
def create_word_embeddings(text_data):
    # Tokenize the text data
    tokenized_data = [tokenize(text) for text in text_data]
    # Train the Word2Vec model on the tokenized data
    model = Word2Vec(tokenized_data, size=100, window=5, min_count=1, workers=4)
    # Extract the word vectors from the trained model
    word_vectors = model.wv
    # Create a dictionary of word embeddings
    embeddings_dict = {}
    for word in word_vectors.vocab:
        embeddings_dict[word] = word_vectors[word]
    return embeddings_dict

# Load the text data from the dataframe
text_data = df['text'].tolist()

# Create Word2Vec embeddings for the text data
embeddings_dict = create_word_embeddings(text_data)

# Define a function to create additional features using Word Embeddings
def create_embedding_features(text_data, embeddings_dict):
    # Tokenize the text data
    tokenized_data = [tokenize(text) for text in text_data]
    # Create a list of lists to hold the embeddings for each text
    embeddings_list = []
    for text in tokenized_data:
        # Create an empty array to hold the embeddings for each word in the text
        text_embeddings = np.zeros(100)
        for word in text:
            # Check if the word is in the embeddings dictionary
            if word in embeddings_dict:
                # Add the word's embedding to the text's embeddings
                text_embeddings += embeddings_dict[word]
        embeddings_list.append(text_embeddings)
    # Convert the embeddings list to a numpy array
    embeddings_array = np.array(embeddings_list)
    # Normalize the embeddings array
    normalized_embeddings = embeddings_array / np.linalg.norm(embeddings_array, axis=1).reshape(-1, 1)
    return normalized_embeddings

# Create additional features using Word Embeddings
embedding_features = create_embedding_features(text_data, embeddings_dict)

# Add the additional features to the dataframe
df['embedding_features'] = embedding_features.tolist()
