import time
import pandas as pd

print("reading data")
data = pd.read_csv("artikler.csv", encoding="utf-16", sep='\t', index_col=0)
print("DONE reading data")

print(data.info())
print(data)


print(data.iloc[60000].Text)