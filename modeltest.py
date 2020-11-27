from re import sub
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from segmentizer import Segmentizer
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import time
from joblib import dump, load
from quotebert import QuoteBERT
import prototype

positives = pd.read_csv('BERTModels/quotes_unsegmentized_politik.bert', index_col=0)
negatives = pd.read_csv('BERTModels/negatives.bert', index_col=0)
positives['label'] = 1
negatives['label'] = 0

X = positives.append(negatives, ignore_index=True)

y = X.label
X = X.drop(columns='label')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# rfc = RandomForestClassifier()
log_res = LogisticRegression(max_iter=10000)
# maybe svm - different kernel
# big_model = load('models/quotemodel_108016')


# rfc.fit(X_train, y_train)
log_res.fit(X_train, y_train)
# big_model.fit(X_train, y_train)


print('Accuracy logistic regression:', log_res.score(X_test, y_test))
# print('Accuracy logistic regression - big model:', big_model.score(X_test, y_test))

# print('Accuracy random forrest classifier:', rfc.score(X_test, y_test))

y_pred = log_res.predict(X_test)
y_prob = log_res.predict_proba(X_test)

predictions = pd.DataFrame(y_test)
predictions['pred'] = y_pred
predictions['proba'] = y_prob.tolist()

# print (predictions)

print('predicting on new data...')
text = Segmentizer.textfile_to_dataframe('data/queen2019.txt').reset_index(drop=True)
text = text.dropna()
qb = QuoteBERT()
X = qb.generate_vectors(text.iloc[:,0].values.tolist())
X = pd.DataFrame(X)
predictions = log_res.predict(X)
text['predict'] = predictions
text['score'] = log_res.predict_proba(X)[:,1]
pd.set_option('display.max_rows', None)

pd.set_option('display.max_colwidth', 100)
print (text[['Quotes', 'predict','score']])

print ('TOP 10 QUOTE CANDIDATES')
print (text.sort_values(by='score', ascending=False).head(10))
print()
print()


print ('BOTTOM 10 QUOTE CANDIDATES')
print (text.sort_values(by='score', ascending=True).head(10))

prototype.to_html(text[['Quotes', 'score']].values.tolist())

plot_confusion_matrix(log_res, X_test, y_test)
plt.show()