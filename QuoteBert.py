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
        
        print(type(data))

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
            # maybe make a list and concatenate in the end
        
        t = time.time()-t
        print()
        print('DONE getting vectors')
        print('Fetched', len(self.X), 'vectors in', t, 'seconds. Average time:', t/len(self.X), 'per sample')

        if save_file == True:
            self.save_to_file(file_name)
    
        print(self.X)
        return self.X

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
    politik = pd.read_csv('largefiles/Quotes_unsegmentized_Politik_36877.tsv', sep='\t', index_col=0).head(10000).iloc[:,0].values.tolist()
    # sport = pd.read_csv('largefiles/Quotes_Sport_92823.tsv', sep='\t', index_col=0).iloc[:,0].values.tolist()
    # kultur = pd.read_csv('largefiles/Quotes_kultur_68104.tsv', sep='\t', index_col=0).iloc[:,0].values.tolist()
    # nyheder = pd.read_csv('largefiles/Quotes_Nyheder_480052.tsv', sep='\t', index_col=0).iloc[:,0].values.tolist()
    # negatives = pd.read_csv('data/ft2016_full.tsv', sep='\t', index_col=0).iloc[:,0].dropna().values.tolist()


    qb = QuoteBERT()
    qb.generate_vectors(politik, save_file=True, sort=True, file_name='BERTModels/quotes_unsegmentized_politik.bert')
    # qb.generate_vectors(nyheder, save_file=True, sort=True, file_name='BERTModels/quotes_nyheder.bert')
    # qb.generate_vectors(sport, save_file=True, sort=True, file_name='BERTModels/quotes_sport.bert')
    # qb.generate_vectors(kultur, save_file=True, sort=True, file_name='BERTModels/quotes_kultur.bert')
    # qb.generate_vectors(negatives, save_file=True, sort=True, file_name='BERTModels/negatives.bert')


    # vec = qb.get_vectors()

