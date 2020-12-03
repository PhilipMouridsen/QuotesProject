import time
import json
import pandas as pd
import matplotlib.pyplot as plt

print("reading data")
quotes = pd.read_csv("largefiles/Quotes_unsegmentized_Nyheder_258502.tsv", encoding="utf-8", sep='\t', index_col=0)
negatives = pd.read_csv("largefiles/ft2016_combined.tsv", encoding="utf-8", sep='\t', index_col=0)
print("DONE reading data")

pd.set_option('display.max_colwidth', -1)

print(quotes.sample(n=10))
print(negatives.sample(n=10))