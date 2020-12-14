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

start_time = time.time() # for timing

# setup BERT
model_class = tf.DistilBertModel
tokenizer_class = tf.DistilBertTokenizer
pretrained_weights = 'distilbert-base-multilingual-cased'
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

elapsed = time.time() - start_time
print('initialized bert in', elapsed, 'seconds')

# load and preprocess the quptes
quotes_file = pd.read_csv("data/quotes10000.csv", encoding='utf-16', sep='\t', index_col=0, converters={'Quotes': eval})
quotes_file = quotes_file.explode('Quotes').drop(columns=['Pub.', 'Text', 'Titel', 'Område', 'Format'])
quotes_file = quotes_file.dropna(subset=['Quotes'])
quotes = quotes_file[['Quotes']]
quotes['Quotes'] = quotes_file.Quotes.apply(Segmentizer.get_segments)
quotes = quotes.explode('Quotes')
quotes = quotes[quotes['Quotes'].map(len) < 400] # drop very long sentences as they are probably due to error in data
quotes = quotes.reset_index(drop=True).sample(10000) # change to change the number of quotes to use for training
quotes['label'] = 1 # assign positive label to quotes

# load and preprocess the non-quotes
negatives = pd.read_csv('data/ft2016.tsv', encoding='utf-8', sep='\t', index_col=0).reset_index(drop=True)
negatives.columns = ['Quotes']
negatives = negatives.head(100000)
negatives['label'] = 0 # assign negative label to quotes

# combine the two types of data
combined = pd.concat([quotes, negatives], axis=0).dropna().reset_index(drop=True)

# split data into batches so the BERT process does not run out of memory
batch_size = 4
batches = [combined[i:min(i + batch_size, len(combined))] for i in range(0, len(combined), batch_size)]

iteration = 0
n_iterations = len(batches)
X = np.empty((0,768))

# loop to process each batch
for batch in batches:
    iteration += 1 # counter
    print (batch.index)

    # transform the sentences to tokens
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

    # do the actual work - and get the precious vectors
    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)
    
    vectors = last_hidden_states[0][:,0,:].numpy()
    
    # add the newly generated batch of vectors to the collection
    X = np.concatenate((X,vectors), axis=0)
    print('Training BERT - iteration', iteration,'/', n_iterations)

print('Finished Training BERT with', len(X), 'strings. Total time:', time.time()-start_time, 'seconds')

# get the lables
y = combined['label']

# split the data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

# train the model
log_res = LogisticRegression(max_iter=10000)
log_res.fit(X_train, y_train)

print('Accuracy:', log_res.score(X_test, y_test))

# save the model to file
filename = 'largemodels/quotemodel_' + str(len(X))
dump(log_res, filename+'.joblib')

# save the vectors to file
df_vectors = pd.DataFrame(X)
df_vectors['label'] = y
df_vectors.to_csv(filename+'.bert', index=False)

# show confusion matrix of result
plot_confusion_matrix(log_res, X_test, y_test)
plt.show()
