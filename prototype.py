import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from segmentizer import Segmentizer
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from quotebert import QuoteBERT
import webbrowser

def score_to_color(score):
    color = ""
    if score > .5:
        hue = (1- (score-.5)*2) * 100
        color = 'hsl(' + str(hue) + ", 50%, 50%)"
    return color

def score_to_color_alpha(score):
    color = ""
    if score > .5:
        lightness = (abs(score-1)+.5)*100
        color = 'hsl(120, 100%,' + str(lightness) + '%)'
    else:
        lightness = (score+0.5)*100
        color = 'hsl(0, 100%,' + str(lightness) + '%)'

    return color


# take a list of (text, score) pairs
# encode each string with a color representing the score
# generate HTML, save to file and open result in browser
def to_html(lst):
    
    body = '<p>'

    head = '<!doctype html><head><meta charset="utf-16"><title>QuoteBERT in Action!</title> \
        <meta name="description" content="Prototype of Quote Selection using BERT"> \
        <meta name="author" content="Lasse Funder Andersen">\
        </head><body><p style="font-family:arial;font-size:16px>'

    closing = '</p></body>'

    for q, s in lst:
        body = body + '<span style=\"background-color:' + score_to_color_alpha(s) +';\">' + q + ' </span>'

    html = head + body + closing
    with open('test.html', 'w', encoding='utf-8') as file:      
        file.write(html)
    
    webbrowser.open('test.html')

positives = pd.read_csv('BERTModels/quotes_unsegmentized_nyheder_99962.bert', index_col=0).sample(n=10000)
# positives = pd.read_csv('BERTModels/quotes_unsegmentized_politik.bert', index_col=0).sample(n=10000)
# positives = pd.read_csv('BERTModels/quotes_unsegmentized_sport_44408.bert', index_col=0).sample(n=10000)


negatives = pd.read_csv('BERTModels/negatives_combined_27081.bert', index_col=0).sample(n=10000)
positives['label'] = 1
negatives['label'] = 0

X = positives.append(negatives, ignore_index=True)

y = X.label
X = X.drop(columns='label')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

log_res = LogisticRegression(max_iter=1000)
log_res.fit(X_train, y_train)

print('Accuracy logistic regression:', log_res.score(X_test, y_test))

y_pred = log_res.predict(X_test)
y_prob = log_res.predict_proba(X_test)

predictions = pd.DataFrame(y_test)
predictions['pred'] = y_pred
predictions['proba'] = y_prob.tolist()


print('predicting on new data...')
text = Segmentizer.textfile_to_dataframe('data/queen2019.txt', make_doubles=False).reset_index(drop=True)
text = text.dropna()
qb = QuoteBERT()
X = qb.generate_vectors(text.iloc[:,0].values.tolist())
X = pd.DataFrame(X)
predictions = log_res.predict(X)
text['predict'] = predictions
text['score'] = log_res.predict_proba(X)[:,1]
pd.set_option('display.max_rows', None)

pd.set_option('display.max_colwidth', None)
# print (text[['Quotes', 'predict','score']])

print ('TOP 10 QUOTE CANDIDATES')
print (text.sort_values(by='score', ascending=False).head(10))
print()
print()


print ('BOTTOM 10 QUOTE CANDIDATES')
print (text.sort_values(by='score', ascending=True).head(10))

to_html(text[['Quotes', 'score']].values.tolist())

plot_confusion_matrix(log_res, X_test, y_test)
plt.show()