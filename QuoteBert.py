import re
import numpy as np
from transformers import BertTokenizer, BertModel
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
from segmentizer import Segmentizer


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


################################################################




# get data
# qt holds quotes from 100 articles
qt = pd.read_csv("data/quotes100.csv", encoding='utf-16', sep='\t', index_col=0, converters={'Quotes': eval})
qt = qt.explode('Quotes').drop(columns=['Pub.', 'HTML', 'Text', 'Titel', 'Område', 'URL', 'Format'])
qt = qt.dropna()
qt['Quotes'] = qt.Quotes.apply(Segmentizer.get_segments)
qt = qt.explode('Quotes')
print (qt)
qt.reset_index(drop=True, inplace=True)

# not_qt holds segments from the danish wikipedia-article on Denmark
not_qt = pd.read_csv("data/wiki-segmentized.csv", sep='\t', index_col=0)
not_qt.columns= ['Quotes']

print(not_qt)

# Settings
model_tag = "bert-base-multilingual-uncased"
cls_loc = 0

# Get a 'tokenizer', it converts words/tokens to token-numbers that represent those words
tokenizer = BertTokenizer.from_pretrained(model_tag)

# Get the BERT model, it takes the tokenized word-numbers and does magic on it (for example get vector output)
model = BertModel.from_pretrained(model_tag)

# prepare data 
def prepare(text):
    return tokenizer(text, padding=True, return_tensors="pt")

def back_to_text(input_ids):
    result = ""
    for i in range(input_ids.shape[0]):
        temp = tokenizer.convert_ids_to_tokens(input_ids[i, :])  # Convert token-numbers back to token-strings
        result = result + " " + ids2text(temp)
    return result

def get_BERT_vector(str):
    prepared = prepare(str)
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

    return vector[0]


qt['vec'] = qt['Quotes'].apply(get_BERT_vector)
not_qt['vec'] = not_qt['Quotes'].apply(get_BERT_vector)
print(qt)


qt['is_quote'] = 1
not_qt['is_quote'] = 0
combined = pd.concat([qt, not_qt]).reset_index()
extracted = pd.DataFrame(combined['vec'].tolist(), index=combined.index)

combined = combined.merge(extracted, right_on=extracted.index, left_on=combined.index)
combined = combined.drop(columns=['Quotes','index','key_0','vec'])

X_train, X_test, y_train, y_test = train_test_split(combined.drop(columns=['is_quote']), combined.is_quote, test_size=0.2)

print (X_train)
print (X_test)

model = SVC()
model.fit(X_train, y_train)

plot_confusion_matrix(model, X_test, y_test)
plt.show()

score = model.score(X_test, y_test)
print("Score:", score )


