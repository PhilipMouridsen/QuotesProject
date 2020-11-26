import numpy as np
import pandas as pd
from segmentizer import Segmentizer

data = ""
with open('data/ft2016.txt', encoding='utf-8') as file:
    data = file.readlines()

sentences=[]
for line in data:
    line = line.replace('\n', '')
    new_sentences = Segmentizer.get_segments(line)
    sentences.extend(new_sentences)


df = pd.DataFrame(sentences)

print(df)

df.to_csv('data/ft2016.tsv', sep='\t', encoding='utf-8')