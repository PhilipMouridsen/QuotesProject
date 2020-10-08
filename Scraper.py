# scraper
import pandas as pd
import urllib.request
from urllib.request import urlopen
from bs4 import BeautifulSoup

data = pd.read_csv("artikelurls.csv", encoding="utf-16", sep='\t')
total = 0

print(data.describe)
print(data.head)

print(data.columns)
data = data.drop(columns=['URN, Pub., Format and 2 more (Combined)','Brugere', 'Gns. tid', 'Afs. video', '% Afs. video','SoMe', '% SoMe', 'Search', '% Search', 'URN Count d' ])
print(data)


def getHTML(url):
    global total
    total = total + 1 
    print("fetching article", total)
    fp = urlopen(url).read()
    print ("DONE")
    return fp
    


def get_text_from_HTML(html):
    soup = BeautifulSoup(html, features="html.parser")
    result = ""
    paragraphs = soup.findAll('p', attrs={'class':'dre-article-body__paragraph'})
    
    newline = ""
    for p in paragraphs:
        result = result + p.text + newline
        newline = "\n"

    return result

data['HTML'] = [getHTML(x) for x in data['URL']]
data['Text'] = [get_text_from_HTML(x) for x in data['HTML']]
data = data.drop(columns=['HTML', 'URL'])

data.to_csv("test.csv", encoding="utf-16", sep='\t')

# test the file
test = pd.read_csv('test.csv',encoding="utf-16", sep='\t')
print(test)

# for text in test['Text']:
#    print(text)

# html = urlopen(url).read()