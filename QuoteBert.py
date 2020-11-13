import re
import numpy as np
from transformers import BertTokenizer, BertModel
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt


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
qt.reset_index(drop=True, inplace=True)

# not_qt holds lines from a speech by the speaker of the danish parliament
not_qt = pd.read_csv("data/tale.txt", sep='\n')
not_qt.columns= ['Quotes']


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
print(combined)
print(extracted)

combined = combined.merge(extracted, right_on=extracted.index, left_on=combined.index)
combined = combined.drop(columns=['Quotes','index','key_0','vec'])
print(combined)

X_train, X_test, y_train, y_test = train_test_split(combined.drop(columns=['is_quote']), combined.is_quote, test_size=0.2)

print (X_train)
print (X_test)

model = SVC()

model.fit(X_train, y_train)
plot_confusion_matrix(model, X_test, y_test)
plt.show()

score = model.score(X_test, y_test)
print("Score:", score )

# Prepare data for model - we use the tokenizer for creating the necessary input for BERT.
# All kinds of cool things can be done here depending on what task we are working on
# prepared_data = tokenizer(texts, padding=True, return_tensors="pt")

# Here I just extract the necessary components
# input_ids = prepared_data["input_ids"]
# token_type_ids = prepared_data["token_type_ids"]
# attention_mask = prepared_data["attention_mask"]
# assert all(input_ids[:, cls_loc] == tokenizer.cls_token_id)  # I'm just making sure its correct so far

# Print data - for sanity
# You can see how BERT uses special tokens like [CLS], [SEP] and [PAD] for various things
# for i in range(input_ids.shape[0]):
#     temp = tokenizer.convert_ids_to_tokens(input_ids[i, :])  # Convert token-numbers back to token-strings
#     print(f"{temp!s:100s}  -->  {ids2text(temp)}")  # Print nicely

# # Run BERT model
# model_output = model(
#     input_ids=input_ids,
#     token_type_ids=token_type_ids,
#     attention_mask=attention_mask,
#     return_dict=True,
# )

# # Get stuff
# last_hidden_state = model_output["last_hidden_state"]
# pooler_output = model_output["pooler_output"]

# # The state at the "[CLS]"-token is what should be used for classification
# classification_vectors = last_hidden_state[:, cls_loc, :].detach().numpy()  # type: np.ndarray
# print(classification_vectors)


# classification_vectors is now a numpy array of shape [n_samples, n_dimensions] which can be used for classification





# Gennemgang af Jeppes eksempel
# Forstå classification-fremgangsmåden
# preprocessing - del op i sætninger? hvor mange negative vs positive eksempler?
# Indhold rapporten
#       - afsnit, hvad forventer han?
#       - disposition
# Sentiment-analyse?
# 0 1 - 0.5 