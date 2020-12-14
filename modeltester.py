import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

from joblib import dump, load
from sklearn.metrics import classification_report

nyheder = pd.read_csv('BERTModels/quotes_unsegmentized_nyheder_99962.bert', index_col=0)
politik = pd.read_csv('BERTModels/quotes_unsegmentized_politik.bert', index_col=0)
sport = pd.read_csv('BERTModels/quotes_unsegmentized_sport_44408.bert', index_col=0)
negatives = pd.read_csv('BERTModels/negatives_combined_27081.bert', index_col=0)

# nyheder = pd.read_csv('BERTModels/quotes_unsegmentized_nyheder_1000.bert', index_col=0).sample(n=10000)
# politik = pd.read_csv('BERTModels/quotes_unsegmentized_politik.bert', index_col=0).sample(n=10000)
# sport = pd.read_csv('BERTModels/quotes_unsegmentized_sport_1000.bert', index_col=0).sample(n=10000)
# negatives = pd.read_csv('BERTModels/negatives_combined_9134.bert', index_col=0).sample(n=10000)


nyheder['label'] = 1
sport['label'] = 1
politik['label'] = 1
negatives['label'] = 0

print('LOADED DATA')

base = 25000

positives_list = {'Nyheder':nyheder, 'Sport':sport, 'Politik':politik}
ratios = [1]
for name, positives in positives_list.items():
    print('POSITIVE DATA:', name)
    for r in ratios:
        print('RATIO:', r)
        print('POSITIVES:',len(positives))
        print('NEGATIVES:',len(negatives))

        X = positives.sample(n=int(base/r)).append(negatives.sample(n=base), ignore_index=True)
        y = X.label
        X = X.drop(columns='label')

        print(len(X))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        log_res = LogisticRegression(max_iter=10000)
        svc = SVC()

        log_res.fit(X_train, y_train)
        svc.fit(X_train, y_train)

        pred_log_res = log_res.predict(X_test)
        pred_svc = svc.predict(X_test)

        print('LOGISTIC REGRESSION')
        print(classification_report(y_test, pred_log_res))
        print('SCORE:', log_res.score(X_test, y_test))

        print('SVC')
        print()
        print()
        print(classification_report(y_test,pred_svc))
        print('SCORE:',svc.score(X_test, y_test))



    # plot_confusion_matrix(log_res, X_test, y_test)
    # plt.show()