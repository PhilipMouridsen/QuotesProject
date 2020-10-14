import time
import json
import pandas as pd
import matplotlib.pyplot as plt

print("reading data")
data = pd.read_csv("data/quotes10000.csv", encoding="utf-16", sep='\t', index_col=0, converters={'Quotes': eval})
print("DONE reading data")

data['nQuotes'] = data.apply(lambda row: len(row.Quotes), axis=1)

print(data.info())
print(data)

hist = data.hist(column='nQuotes', bins=50)
plt.show()
 


nQuotes = 0
articles_with_quotes = 0
for quotes in data['Quotes']:
    nQuotes = nQuotes + len(quotes)
    if len(quotes) > 0: articles_with_quotes += 1

print("Number of articles:")
print(articles_with_quotes)
print("number of quotes:")
print (nQuotes)
print()
print ("avg", nQuotes/articles_with_quotes)