import pandas as pd
url= 'https://filedn.eu/lefeldrXcsSFgCcgc48eaLY/datasets/clusteringOFF/en.openfoodfacts.org.products.csv'
df = pd.read_csv(url, sep='\t', low_memory=False)
print(df.head(5))