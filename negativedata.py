import numpy as np
import pandas as pd
from segmentizer import Segmentizer

data = ""
with open('data/ft2016.txt', encoding='utf-8') as file:
    data = file.readlines()

print (len(data))
sentences=[]
for line in data:
    line = line.replace('\n', '')
    new_sentences = Segmentizer.get_segments(line)
    sentences.extend(new_sentences)

df = pd.DataFrame(sentences)
print(len(df))

previous_sentence = ""
combined_sentences = []
for s in sentences:
        combined_sentence = previous_sentence + ' ' + s
        # print (combined_sentence)
        combined_sentences.append(combined_sentence)
        previous_sentence = s

# print(combined_sentences)
total = sentences+combined_sentences

df = pd.DataFrame(total).dropna().reset_index(drop=True)
print(len(df))
print(df)

df.to_csv('largefiles/ft2016_combined.tsv', sep='\t', encoding='utf-8')