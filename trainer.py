import numpy as np
import pandas as pd
import torch
import transformers as tf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from segmentizer import Segmentizer
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import time
from joblib import dump, load

start_time = time.time()

# setup BERT
model_class = tf.DistilBertModel
tokenizer_class = tf.DistilBertTokenizer
pretrained_weights = 'distilbert-base-multilingual-cased'
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

elapsed = time.time() - start_time
print('initialized bert in', elapsed, 'seconds')

quotes_file = pd.read_csv("data/quotes10000.csv", encoding='utf-16', sep='\t', index_col=0, converters={'Quotes': eval})
quotes_file = quotes_file.head(100)
quotes_file = quotes_file.explode('Quotes').drop(columns=['Pub.', 'Text', 'Titel', 'Område', 'Format'])
quotes_file = quotes_file.dropna(subset=['Quotes'])
quotes = quotes_file[['Quotes']]
quotes['Quotes'] = quotes_file.Quotes.apply(Segmentizer.get_segments)
quotes = quotes.explode('Quotes')
quotes.reset_index(drop=True, inplace=True)
quotes['label'] = 1

negatives = pd.read_csv('data/ft2016.tsv', encoding='utf-8', sep='\t', index_col=0)
negatives.columns = ['Quotes']
negatives.dropna(subset=['Quotes'])
negatives = negatives.head(1000)
negatives['label'] = 0

print(negatives)

combined = pd.concat([quotes, negatives], axis=0).reset_index(drop=True)

elapsed = time.time()-elapsed
# print('loaded data in', elapsed, 'seconds')

print (combined)

# split data into batches
batch_size = 4
batches = [combined[i:min(i + batch_size, len(combined))] for i in range(0, len(combined), batch_size)]

iteration = 0
n_iterations = len(batches)
X = np.empty((0,768))
for batch in batches:
    iteration += 1
    tokenized = batch['Quotes'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

    # add padding for uniform length
    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)

    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

    attention_mask = np.where(padded != 0, 1, 0)
    input_ids = torch.tensor(padded).to(torch.long)  
    attention_mask = torch.tensor(attention_mask)

    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)

    print('Training BERT - iteration', iteration,'/', n_iterations)

    vectors = last_hidden_states[0][:,0,:].numpy()
    X = np.concatenate((X,vectors), axis=0)

print('Finished Training BERT with', len(X), 'strings. Total time:', time.time()-start_time, 'seconds')

y = combined['label']

df_vectors = pd.DataFrame(X)
df_vectors['label'] = y

print(df_vectors)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

log_res = LogisticRegression(max_iter=10000)
log_res.fit(X_train, y_train)

print('Accuracy:', log_res.score(X_test, y_test))


filename = 'models/quotemodel_' + str(len(X))

dump(log_res, filename+'.joblib')
df_vectors.to_csv(filename+'.bert', index=False)

plot_confusion_matrix(log_res, X_test, y_test)
plt.show()
