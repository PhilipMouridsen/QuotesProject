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

X = pd.read_csv('models/quotemodel_1384.bert')

print(X)
y = X['label']
X = X.drop(columns=['label'])

print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


rfc = RandomForestClassifier()
log_res = LogisticRegression(max_iter=10000)

rfc.fit(X_train, y_train)
log_res.fit(X_train, y_train)

print('Accuracy logistic regression:', log_res.score(X_test, y_test))
print('Accuracy random forrest classifier:', rfc.score(X_test, y_test))

y_pred = log_res.predict(X_test)
y_prob = log_res.predict_proba(X_test)

predictions = pd.DataFrame(y_test)
predictions['pred'] = y_pred
predictions['proba'] = y_prob.tolist()

print (predictions)


plot_confusion_matrix(log_res, X_test, y_test)
plt.show()

