import time
from matplotlib.pyplot import xlabel, xticks
import pandas as pd
import numpy as np
import re

from segmentizer import Segmentizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from afinn import Afinn



# load and preprocess the quotes
quotes = pd.read_csv("largefiles/quotesAll.csv", encoding='utf-16', sep='\t', index_col=0, converters={'Quotes': eval})
quotes = quotes.explode('Quotes').drop(columns=['Pub.', 'Text', 'Titel', 'Format', 'HTML', 'URL'])
quotes = quotes.dropna(subset=['Quotes'])
quotes['Quotes'] = quotes.Quotes.apply(Segmentizer.get_segments)

# quotes = quotes.explode('Quotes')


pattern = r', '
# quotes['Område'] = quotes['Område'].apply(lambda x: str.split(x, ', '))
# quotes = quotes.explode('Område')
# quotes['Område'] = quotes['Område'].apply(lambda x: str.split(x, ','))
# quotes = quotes.explode('Område')
# quotes = quotes.dropna(subset=['Quotes'])

pd.set_option('display.max_rows', 10)

# print (quotes.groupby('Område').count().sort_values(by='Quotes',ascending=False))

quotes['n_quotes'] = quotes.Quotes.apply(len)

# print (quotes)

nyheder = pd.read_csv("largefiles/Quotes_unsegmentized_Nyheder_258502.tsv", encoding='utf-8', sep='\t', index_col=0).sample(n=1000)
politik = pd.read_csv("largefiles/Quotes_unsegmentized_Politik_36877.tsv", encoding='utf-8', sep='\t', index_col=0).sample(n=1000)
negatives = pd.read_csv("largefiles/ft2016_combined.tsv", encoding='utf-8', sep='\t', index_col=0).dropna().sample(n=1000)
kultur = pd.read_csv("largefiles/Quotes_unsegmentized_Nyheder_258502.tsv", encoding='utf-8', sep='\t', index_col=0).sample(n=1000)
sport = pd.read_csv("largefiles/Quotes_unsegmentized_Nyheder_258502.tsv", encoding='utf-8', sep='\t', index_col=0).sample(n=1000)


quotes['Segments'] = quotes.Quotes.apply(len)

print (nyheder)

plt.hist(quotes['Segments'], bins=6)
plt.title('Distribution of number of segments in all quotes')
plt.xlabel('Number of Segments')
plt.ylabel('n')
plt.show()

senti = Afinn(language='da')
def get_sentiment(tweet):
    s = senti.score(tweet)
    return s


nyheder['Sentiment'] = nyheder.Quotes.apply(get_sentiment)
negatives['Sentiment'] = negatives['0'].apply(get_sentiment)
sport['Sentiment'] = sport.Quotes.apply(get_sentiment)
kultur['Sentiment'] = kultur.Quotes.apply(get_sentiment)
politik['Sentiment'] = politik.Quotes.apply(get_sentiment)


plt.hist(negatives.Sentiment, bins=negatives.Sentiment.nunique(), label='Negatives',histtype='step')
plt.hist(politik.Sentiment, bins=politik.Sentiment.nunique(), label='Politik Quotes',histtype='step')
plt.hist(kultur.Sentiment, bins=kultur.Sentiment.nunique(), label='Kultur Quotes',histtype='step')
plt.hist(sport.Sentiment, bins=sport.Sentiment.nunique(), label='Sport Quotes',histtype='step')
plt.hist(nyheder.Sentiment, bins=nyheder.Sentiment.nunique(), label='Nyheder Quotes',histtype='step')
plt.xlabel('Afinn sentiment score')
plt.ylabel('n')
plt.title('Distribution of Sentiment')
plt.legend()
plt.show()

n, bins, patches = plt.hist(x=quotes.n_quotes, bins=np.arange(19) - 0.5, color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.title('Distribution of the number of quotes in each article')
plt.xlabel('Number of quotes in article')
plt.ylabel('n')
plt.xticks(np.arange(19))
plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
# plt.ylim(top=15000)
plt.xlim(right=8)

plt.show()

negatives = pd.read_csv("BERTModels/negatives_combined_27081.bert", encoding='utf-8').sample(n=1000)
kultur = pd.read_csv("BERTModels/quotes_unsegmentized_kultur_32842.bert", encoding='utf-8').sample(n=1000)
politik = pd.read_csv("BERTModels/quotes_unsegmentized_politik.bert", encoding='utf-8').sample(n=1000)
sport = pd.read_csv("BERTModels/quotes_unsegmentized_sport_44408.bert", encoding='utf-8').sample(n=1000)
nyheder = pd.read_csv("BERTModels/quotes_unsegmentized_nyheder_99962.bert", encoding='utf-8').sample(n=1000)


# Plot 1000 examples using PCA and TSNE
kultur_reduced = pd.DataFrame(TSNE(n_components=2).fit_transform(PCA(n_components=50).fit_transform(kultur)))
politik_reduced = pd.DataFrame(TSNE(n_components=2).fit_transform(PCA(n_components=50).fit_transform(politik)))
sport_reduced = pd.DataFrame(TSNE(n_components=2).fit_transform(PCA(n_components=50).fit_transform(sport)))
negatives_reduced = pd.DataFrame(TSNE(n_components=2).fit_transform(PCA(n_components=50).fit_transform(negatives)))
nyheder_reduced = pd.DataFrame(TSNE(n_components=2).fit_transform(PCA(n_components=50).fit_transform(nyheder)))

plt.scatter(negatives_reduced[0], negatives_reduced[1], alpha=0.5, label='Negatives')
plt.scatter(politik_reduced[0], politik_reduced[1], alpha=0.5, label='Politik')
plt.scatter(kultur_reduced[0], kultur_reduced[1], alpha=0.5, label='Kultur')
plt.scatter(sport_reduced[0], sport_reduced[1], alpha=0.5, label='Sport')
plt.scatter(nyheder_reduced[0], nyheder_reduced[1], alpha=0.5, label='Nyheder')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Plot of BERT vectors after dimensionality reduction using t-SNE')


plt.legend()
plt.show()

# Plot 1000 examples using PCA
kultur_reduced = pd.DataFrame(PCA(n_components=2).fit_transform(kultur))
politik_reduced = pd.DataFrame(PCA(n_components=2).fit_transform(politik))
sport_reduced = pd.DataFrame(PCA(n_components=2).fit_transform(sport))
negatives_reduced = pd.DataFrame(PCA(n_components=2).fit_transform(negatives))
nyheder_reduced = pd.DataFrame(PCA(n_components=2).fit_transform(nyheder))

plt.scatter(negatives_reduced[0], negatives_reduced[1], alpha=0.5, label='Negatives')
plt.scatter(politik_reduced[0], politik_reduced[1], alpha=0.5, label='Politik')
plt.scatter(kultur_reduced[0], kultur_reduced[1], alpha=0.5, label='Kultur')
plt.scatter(sport_reduced[0], sport_reduced[1], alpha=0.5, label='Sport')
plt.scatter(nyheder_reduced[0], nyheder_reduced[1], alpha=0.5, label='Nyheder')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Plot of BERT vectors after dimensionality reduction with PCA')
plt.legend()
plt.show()



