from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from typing import Union
import pandas as pd

class Embeding_visualizer():
    """Class to vizualize embeding with t-SNE"""
    def __init__(self):
        self.model = None

    def __setattr__(self, __name: str, __value) -> None:
        """Override the __setattr__ method to check the type of the attribute
            args :
            __name (str) : name of the attribute
            __value : value of the attribute
        """
        if __name == "model" and not isinstance(__value, Word2Vec) and __value is not None:
            raise TypeError("model must be a Word2Vec instance")
        super().__setattr__(__name, __value)


    def GenerateToken(self, text: Union[str, list]) -> list:
        """Convert text to tokens
            args : 
            text (str or list)
                The text or list of word to convert
            return : tokens (list) or None
        """
        if (not isinstance(text, str) and not isinstance(text, list)) or text is None:
            raise TypeError("text must be a str or a list")
        # If the text is a str, convert it to a list of str
        if isinstance(text, str):
            text = [text.lower().split(" ")]
        tokens = []
        for sentence in text:
            # Convertir les phrases en liste de mots
            token = simple_preprocess(sentence)
            # Remove the stop words
            stop_words_french = stopwords.words('french')
            stop_words_english = stopwords.words('english')
            token = [word for word in token if word not in stop_words_french and word not in stop_words_english]
            tokens.append(token)

        return tokens


    def trainModel(self, tokens: list, **kwargs) -> Union[list, None]:
        """Train the model
            args :
                tokens (list) :
                    The list of tokens
                **kwargs :
                    The arguments of the Word2Vec model
            return :
                list of the words in the model
        """
        if not isinstance(tokens, list):
            raise TypeError("tokens must be a list")
        # Train the model
        self.model = Word2Vec(sentences=tokens, vector_size=100, window=5, min_count=1, workers=4, **kwargs)
        self.model.train(tokens, total_examples=len(tokens), epochs=10)
        return self.model.wv.key_to_index


    def getMostSimilarWords(self, word: str, topn: int = 10, similar : bool = True, 
                            return_only_word : bool = True) -> list:
        """Get the most similar words
            args :
                word (str) :
                    The word to compare
                topn (int) :
                    The number of words to return
                similar (bool) :
                    If True, return the most similar words
                    If False, return the most dissimilar words
                return_only_word (bool) :
                    If True, return only the words
                    If False, return the words and the score
            return :
                list of the most similar words
        """
        if self.model is None:
            raise ValueError("model is None, please train the model before using trainModel method")
        if not isinstance(word, str):
            raise TypeError("word must be a str")
        if not isinstance(topn, int):
            raise TypeError("topn must be an int")
        if not isinstance(similar, bool):
            raise TypeError("similar must be a bool")
        
        if similar:
            result = self.model.wv.most_similar(word, topn=topn)
        else:
            result = self.model.wv.most_similar(negative=[word], topn=topn)
        if return_only_word:
            return [return_word[0] for return_word in result]
        else:
            return result


    def getWordVector(self, word: str) -> list:
        """Get the vector of a word
            args :
                word (str) :
                    The word to compare
            return :
                list of the vector of the word
        """
        if self.model is None:
            raise ValueError("model is None, please train the model before using trainModel method")
        if not isinstance(word, str):
            raise TypeError("word must be a str")
        
        return self.model.wv[word]
    

    def get_tsne_transform(self, datas : list, perplexity: float = 40, 
                           n_components: int = 2, init: str = "pca", 
                           n_iter: int = 2500):
        """Creates and TSNE model and plots it
            args :
                datas (pd.DataFrame) :
                    The datas to transform
                perplexity (int) : 
                    The perplexity is related to the number of nearest neighbors 
                    that is used in other manifold learning algorithms. 
                    Larger datasets usually require a larger perplexity. 
                    Consider selecting a value between 5 and 50. 
                n_components (int) :
                    Dimension of the embedded space.
                init (str) :
                    Initialization of embedding. 
                    Possible options are ‘random’, ‘pca’, 
                    and a numpy array of shape (n_samples, n_components).
                    PCA initialization cannot be used with precomputed distances
                    and is usually more globally stable than random initialization.
                n_iter (int) :
                    Maximum number of iterations for the optimization.
                return :
                    tsne_model (TSNE) : The tsne model
                    new_values (list) : The new values
        """
        if self.model is None:
            raise ValueError("model is None, please train the model before using trainModel method")

        # Check the type of the arguments
        if not isinstance(datas, list):
            raise TypeError("datas must be a list")
        if not isinstance(perplexity, float):
            raise TypeError("perplexity must be a float")
        if not isinstance(n_components, int):
            raise TypeError("n_components must be an int")
        if not isinstance(init, str):
            raise TypeError("init must be a str")
        if not isinstance(n_iter, int):
            raise TypeError("n_iter must be an int")

        # Create a tsne model and plot it
        word_vector = []
        # Add all tokens to the list
        for word in datas:
            word_vector.append(self.model.wv[word])
        # Create a tsne model
        tsne_model = TSNE(perplexity=perplexity, n_components=n_components, 
                          init=init, n_iter=n_iter, random_state=23)
        # Calculate the new values
        new_values = tsne_model.fit_transform(pd.DataFrame(word_vector).values)

        return tsne_model, new_values

    def plot_tsne(self, tsne_values: list, figsize: tuple = (16, 16)):
        """Plot the tsne values
            args :
                tsne_values (list) :
                    The tsne values
                figsize (tuple) :
                    The size of the figure
        """
        if self.model is None:
            raise ValueError("model is None, please train the model before using trainModel method")

        # Create a tsne model and plot it
        labels = []
        # Add all tokens to the list
        for word in self.model.wv.index_to_key:
            labels.append(word)

        # Keep a track of x y for the annotation
        # with x the component 1 and y the component 2
        x = []
        y = []
        for value in tsne_values:
            x.append(value[0])
            y.append(value[1])

        # Plot the datas    
        plt.figure(figsize=figsize)
        for i in range(len(x)):
            plt.scatter(x[i],y[i])
            # Annotate the points
            plt.annotate(labels[i],
                        xy=(x[i], y[i]),
                        xytext=(5, 2),
                        textcoords='offset points',
                        ha='right',
                        va='bottom')
        plt.show()
    

if __name__ == "__main__":
    # Exemple
    df = pd.read_csv("./data/en.openfoodfacts.org.products.csv", sep="\t", nrows=1000)
    # fill all the nan values with "unknown"
    df = df.fillna("unknown")
    # Create a list of all the ingredients
    ingredients = df["ingredients_text"].values.tolist()
    ev = Embeding_visualizer()
    # Generate the tokens
    tokens = ev.GenerateToken(ingredients)
    # Train the model
    trainedWord = ev.trainModel(tokens)
    # Get the most similar words
    tsne_model, new_values = ev.get_tsne_transform(list(trainedWord.keys()), perplexity=10.0, n_components=2)
    # Plot the tsne values
    ev.plot_tsne(new_values, figsize=(16, 16))