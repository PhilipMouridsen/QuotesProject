import numpy as np
import pandas as pd
import torch
import transformers as tf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from segmentizer import Segmentizer
import time

class QuoteBERT:

    def __init__(self):
        # setup BERT
        print ('Initializing BERT... ', end='')
        self.model_class = tf.DistilBertModel
        self.tokenizer_class = tf.DistilBertTokenizerFast
        self.pretrained_weights = 'distilbert-base-multilingual-cased'
        self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_weights)
        self.model = self.model_class.from_pretrained(self.pretrained_weights)
        self.X = np.empty((0,768))
        print('DONE.')


    # takes a list of strings and generates BERT vectors for each.
    # The resulting list of vectors is stored in QuoteBert object and optionally saved to file
    # where it can be retrieved using pandas.from_csv()
    def generate_vectors(self, data, save_file=False, file_name='largefiles/model.bert', sort=False):
        
        # sorting the strings by length speeds up the tokenization
        if sort == True: data = sorted(data, key=len)
        self.X = np.empty((0,768))

        t = time.time()

        # split data into batches so the BERT process does not run out of memory
        batch_size = 4
        batches = [data[i:min(i + batch_size, len(data))] for i in range(0, len(data), batch_size)]

        iteration = 0
        n_iterations = len(batches)
        
        # loop to process each batch
        for batch in batches:
            iteration += 1 # counter
            print('Training BERT - iteration', iteration,'/', n_iterations, end='\r')

            # transform the sentences to tokens
            tokenized = list(map(lambda x: self.tokenizer.encode(x, add_special_tokens=True), batch))

            # add padding for uniform lengths
            max_len = 0
            for i in tokenized:
                if len(i) > max_len:
                    max_len = len(i)

            padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized])

            # do the actual work - and get the precious vectors
            attention_mask = np.where(padded != 0, 1, 0)
            input_ids = torch.tensor(padded).to(torch.long)  
            attention_mask = torch.tensor(attention_mask)

            with torch.no_grad():
                last_hidden_states = self.model(input_ids, attention_mask=attention_mask)
            
            vectors = last_hidden_states[0][:,0,:].numpy()
            
            # add the newly generated batch of vectors to the collection
            self.X = np.concatenate((self.X,vectors), axis=0)
        
        t = time.time()-t
        print()
        print('DONE getting vectors')
        print('Fetched', len(self.X), 'vectors in', t, 'seconds. Average time:', t/len(self.X), 'per sample')

        if save_file == True:
            self.save_to_file(file_name)

# return the vectors
    def get_vectors(self):
        return self.X

# save vectors to file as a Pandas Dataframe
    def save_to_file(self, filename):
        df = pd.DataFrame(self.X)
        df.to_csv(filename)
        print('BERT vectors saved to ', filename)


# main for testing
if __name__ == '__main__':
    data = pd.read_csv('data/ft2016.tsv', sep='\t', index_col=0)
    data = data.dropna().head(10000)
    print(data)
    data = data.iloc[:,0].values.tolist()
    # print(data)
    qb = QuoteBERT()
    qb.generate_vectors(data, save_file=True, sort=True, file_name='largemodels/ft2016_10000.bert')
    vec = qb.get_vectors()

