import time
import pandas as pd

print("reading data")
data = pd.read_csv("data/quotes10000.csv", encoding="utf-16", sep='\t', index_col=0)
print("DONE reading data")

print(data.info())
print(data)


print(data.iloc[500].Text)
quotes = data.iloc[500].Quotes
print (quotes)