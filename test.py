import time
import json
import pandas as pd

print("reading data")
data = pd.read_csv("largefiles/quotesALL.csv", encoding="utf-16", sep='\t', index_col=0, converters={'Quotes': eval})
print("DONE reading data")

print(data.info())
print(data)


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