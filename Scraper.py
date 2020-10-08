# scraper
import pandas as pd
import urllib.request
from urllib.request import urlopen
from bs4 import BeautifulSoup


data = pd.read_csv("artikelurls10.csv", encoding="utf-16", sep='\t')




print(data.describe)
print(data.head)

print(data.columns)
data = data.drop(columns=['URN, Pub., Format and 2 more (Combined)','Brugere', 'Gns. tid', 'Afs. video', '% Afs. video','SoMe', '% SoMe', 'Search', '% Search', 'URN Count d' ])
print(data)


def getHTML(url):
    print("fetching article")
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
    # get text
    text = soup.get_text()
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return result

data['HTML'] = [getHTML(x) for x in data['URL']]
data['Text'] = [get_text_from_HTML(x) for x in data['HTML']]

print (data)
print (data['URL'].iloc[0])
print (data['Text'].iloc[0])




# html = urlopen(url).read()