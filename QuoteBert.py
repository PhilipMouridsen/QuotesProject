import re
import numpy as np
from transformers import BertTokenizer, BertModel, BertTokenizerFast
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from segmentizer import Segmentizer
import time

################################################################

def ids2text(tokens, concat=True):
    """
    This function simply takes the de-tokenized word from BERT and converts to string.

    :param tokens:
    :param concat:
    :return: str
    """
    single_symbol_pattern = re.compile(r"^\W$")

    # First token
    answer = [tokens[0]]

    # Select the remaining answer tokens and join them with whitespace.
    for token in tokens[1:]:

        # If it's a subword token, then recombine it with the previous token.
        if token[0:2] == '##':
            answer.append(token[2:])

        # Otherwise, add the token with possible space.
        else:
            if single_symbol_pattern.match(token):
                answer.append(token)
            else:
                answer.append(' ' + token)

    # Format
    if concat:
        answer = "".join(answer)

    return answer


# takes a column with values in a list and returns a dataframe with one value in each column 
def list_to_dataframe(column):
    df = pd.DataFrame(column.tolist())
    return df

################################################################




# get data
# qt holds quotes from 100 articles
print("Loading Quotes...")
qt = pd.read_csv("data/quotes100.csv", encoding='utf-16', sep='\t', index_col=0, converters={'Quotes': eval})
qt = qt.explode('Quotes').drop(columns=['Pub.', 'HTML', 'Text', 'Titel', 'Område', 'URL', 'Format'])
qt = qt.dropna()

 # split quotes into segements split by . (punktum)
qt['Quotes'] = qt.Quotes.apply(Segmentizer.get_segments)
qt = qt.explode('Quotes')
qt.reset_index(drop=True, inplace=True)


# Load the negative examples
# not_qt holds segments from the danish wikipedia-article on Denmark
print("Loading Non-Quotes")
not_qt = pd.read_csv("data/wiki-segmentized.csv", sep='\t', index_col=0)
not_qt.columns= ['Quotes']


# Settings
print ("Configuring BERT model...")
model_tag = "bert-base-multilingual-uncased"
cls_loc = 0

# Get a 'tokenizer', it converts words/tokens to token-numbers that represent those words
tokenizer = BertTokenizerFast.from_pretrained(model_tag)

# Get the BERT model, it takes the tokenized word-numbers and does magic on it (for example get vector output)
model = BertModel.from_pretrained(model_tag)

# prepare data 
def prepare(text):
    return tokenizer(text, padding=True, return_tensors="pt")

# for sanity
def back_to_text(input_ids):
    result = ""
    for i in range(input_ids.shape[0]):
        temp = tokenizer.convert_ids_to_tokens(input_ids[i, :])  # Convert token-numbers back to token-strings
        result = result + " " + ids2text(temp)
    return result

# takes a string as input - returns a BERT-vector
def get_BERT_vectors(strings):
    prepared = tokenizer(strings, padding=True, return_tensors="pt")
    input_ids = prepared["input_ids"]
    token_type_ids = prepared["token_type_ids"]
    attention_mask = prepared["attention_mask"]

    model_output = model(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )

    last_hidden_state = model_output["last_hidden_state"]
    vector = last_hidden_state[:, cls_loc, :].detach().numpy()  # type: np.ndarray

    return vector








# convert the strings to BERT vectors in the two tables
print("Assign vectors to quotes")



quotes = qt['Quotes'].to_list()  # Get all texts (the .apply-function is quite limited)
not_quotes = not_qt['Quotes'].to_list() 
batch_size = 64  # You should probably use something like 32 or 64

# Split texts into batches
quote_batches = [quotes[i:min(i + batch_size, len(quotes))] for i in range(0, len(quotes), batch_size)]
not_quote_batches = [not_quotes[i:min(i + batch_size, len(not_quotes))] for i in range(0, len(not_quotes), batch_size)]

# Compute all vectors through BERT

start = time.time()
quote_vectors = np.concatenate([get_BERT_vectors(val) for val in quote_batches], axis=0)
not_quote_vectors = np.concatenate([get_BERT_vectors(val) for val in not_quote_batches], axis=0)
print('Time: ', time.time()-start)

exit()

# qt['vec'] = qt['Quotes'].apply(get_BERT_vector)
# print ('Assign vectors to non-quotes')
# not_qt['vec'] = not_qt['Quotes'].apply(get_BERT_vector)
# qt['is_quote'] = 1
# not_qt['is_quote'] = 0

# combine quotes and non-quotes and get the features and labels for training
combined = pd.concat([qt, not_qt]).reset_index()
y = combined.is_quote
X = list_to_dataframe(combined['vec'])
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

# train classifier on data
svm_clf = SVC()
svm_clf.fit(X_train, y_train)

# show result
plot_confusion_matrix(svm_clf, X_test, y_test)
plt.show()
score = svm_clf.score(X_test, y_test)
print("Score:", score )

##########################3
# test classifier on queens speech
# 
print('predicting on the queens speech...')
queen = Segmentizer.textfile_to_dataframe('data/queen2019.txt').reset_index()
queen['vec'] = queen['Quotes'].apply(get_BERT_vector)
X = list_to_dataframe(queen['vec'])
predictions = svm_clf.predict(X)
queen['predict'] = predictions
pd.set_option('display.max_rows', None)
print (queen[['Quotes', 'predict']].head(len(queen)))



