import re

import numpy as np
from transformers import BertTokenizer, BertModel
from transformers import BertForSequenceClassification
import torch

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

# Settings
model_tag = "bert-base-multilingual-cased"
cls_loc = 0

# Get a 'tokenizer', it converts words/tokens to token-numbers that represent those words
tokenizer = BertTokenizer.from_pretrained(model_tag)

# Get the BERT model, it takes the tokenized word-numbers and does magic on it (for example get vector output)
model = BertForSequenceClassification.from_pretrained(model_tag)

# Some text for testing
texts = [
    "This is a text in English",
    "Den her sætning er på dansk",
    "Jag talar inte svenska",
]

quotelabels = torch.from_numpy(np.array([0, 0, 1])).to(torch.int64)

# Prepare data for model - we use the tokenizer for creating the necessary input for BERT.
# All kinds of cool things can be done here depending on what task we are working on
prepared_data = tokenizer(texts, padding=True, return_tensors="pt")

# Here I just extract the necessary components
input_ids = prepared_data["input_ids"]
token_type_ids = prepared_data["token_type_ids"]
attention_mask = prepared_data["attention_mask"]
assert all(input_ids[:, cls_loc] == tokenizer.cls_token_id)  # I'm just making sure its correct so far

# Print data - for sanity
# You can see how BERT uses special tokens like [CLS], [SEP] and [PAD] for various things
for i in range(input_ids.shape[0]):
    temp = tokenizer.convert_ids_to_tokens(input_ids[i, :])  # Convert token-numbers back to token-strings
    print(f"{temp!s:100s}  -->  {ids2text(temp)}")  # Print nicely

# Run BERT model
model_output = model(labels=quotelabels,
    input_ids=input_ids,
    token_type_ids=token_type_ids,
    attention_mask=attention_mask,
    return_dict=True,
)

# Get stuff
last_hidden_state = model_output["last_hidden_state"]
pooler_output = model_output["pooler_output"]

# The state at the "[CLS]"-token is what should be used for classification
classification_vectors = last_hidden_state[:, cls_loc, :].detach().numpy()  # type: np.ndarray
print(classification_vectors)

# train sklearn classifier



# classification_vectors is now a numpy array of shape [n_samples, n_dimensions] which can be used for classification

# optimizing pytorch
# tutorial on pytorch optimization - the loss

# problmem
    # the data
    # what is a quote
    # quote extraction
    # different kind of quotes - crime, culture, etc

# state of the art
# intended approach
# prototype result

# Quotes 10%
# NonQuotes 90%
# Extract sentences.
# convert to vectors
# add lables - 0,1
# split into train and test - train_test_split from sklearn?
# train the classifier
# test the classifier
# show performance on unknown raw text (queens speech)

# quote extraction
# research. Look beyond NLP. Journalism, PR etc
# 