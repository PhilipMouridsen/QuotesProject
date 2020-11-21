import numpy as np
import pandas as pd
import torch
import transformers as tf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from segmentizer import Segmentizer

class QuoteBERT:

    def __init__(self, data):
        # setup BERT
        self.model_class = tf.DistilBertModel
        self.tokenizer_class = tf.DistilBertTokenizer
        self.pretrained_weights = 'distilbert-base-multilingual-cased'
        self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_weights)
        self.model = self.model_class.from_pretrained(self.pretrained_weights)

        self.data = data
        self.X = np.empty((0,768))
        self.train()

    def train(self):

        # split data into batches so the BERT process does not run out of memory
        batch_size = 4
        batches = [self.data[i:min(i + batch_size, len(self.data))] for i in range(0, len(self.data), batch_size)]

        iteration = 0
        n_iterations = len(batches)
        
        # loop to process each batch
        for batch in batches:
            iteration += 1 # counter
            print (batch.index)

            # transform the sentences to tokens
            tokenized = batch['Quotes'].apply((lambda x: self.tokenizer.encode(x, add_special_tokens=True)))

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
                last_hidden_states = self.model(input_ids, attention_mask=attention_mask)
            
            vectors = last_hidden_states[0][:,0,:].numpy()
            
            # add the newly generated batch of vectors to the collection
            self.X = np.concatenate((self.X,vectors), axis=0)
            print('Training BERT - iteration', iteration,'/', n_iterations)

    def get_vectors(self):
        return self.X


        
        
if __name__ == '__main__':
    data = pd.DataFrame(["Dette er en sætning", "dette er en super fed sætning"])
    data.columns=['Quotes']
    print (data)
    qb = QuoteBERT(data)
    vec = qb.get_vectors()
    print(vec)