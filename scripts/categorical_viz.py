from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


if __name__ == "__main__":
    #data = get_data(file_path = "../data/en.openfoodfacts.org.products.csv", nrows=50)
    #print(f"data set shape is {data.shape}")

    stopwords = set(STOPWORDS)

    cloud = WordCloud(background_color="white", max_words=200, mask=None, 
            stopwords=stopwords, min_font_size=6, width=800, height=400)
    
    cloud.generate('zaza zazez')
    plt.axis("off")
    plt.imshow(cloud, interpolation="bilinear")
    plt.show()