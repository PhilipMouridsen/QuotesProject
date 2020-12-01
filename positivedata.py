import time
import pandas as pd
import re
from segmentizer import Segmentizer

# load and preprocess the quptes
quotes = pd.read_csv("largefiles/quotesAll.csv", encoding='utf-16', sep='\t', index_col=0, converters={'Quotes': eval})
quotes = quotes.explode('Quotes').drop(columns=['Pub.', 'Text', 'Titel', 'Format', 'HTML', 'URL'])
quotes = quotes.dropna(subset=['Quotes'])
# quotes['Quotes'] = quotes.Quotes.apply(Segmentizer.get_segments)
# quotes = quotes.explode('Quotes')


pattern = r', '
quotes['Område'] = quotes['Område'].apply(lambda x: str.split(x, ', '))
quotes = quotes.explode('Område')
quotes['Område'] = quotes['Område'].apply(lambda x: str.split(x, ','))
quotes = quotes.explode('Område')
quotes = quotes.dropna(subset=['Quotes'])

pd.set_option('display.max_rows', None)

print (quotes.groupby('Område').count().sort_values(by='Quotes'))

pd.set_option('display.max_rows', 10)

genres = ['Nyheder', 'Indland', 'Udland', 'Sport', 'Kultur', 'Regionale', 'Politik', 'Penge']

for genre in genres:
    q = quotes[quotes['Område'] == genre]
    print(q)
    filename = 'largefiles/Quotes_unsegmentized_' + genre + "_" + str(len(q.index))+'.tsv'
    q = q.drop(columns=['Område']).reset_index(drop=True)
    print(q)
    print (filename)
    q.to_csv(filename, sep='\t', encoding='utf-8')
