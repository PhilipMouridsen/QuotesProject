# scraper
import time
import pandas as pd
import urllib.request
from urllib.request import urlopen
from bs4 import BeautifulSoup
import concurrent.futures
MAX_THREADS = 100
start_time = time.time()

data = pd.read_csv("data/artikelurls100.csv", encoding="utf-16", sep='\t', index_col=0)
total = 0 # counter for keeping track of progress

# drop unnescessary columns
data = data.drop(columns=['Brugere', 'Gns. tid', 'Afs. video', '% Afs. video','SoMe', '% SoMe', 'Search', '% Search', 'URN Count d' ])


def getHTML(url):
    # getting the raw HTML from a given url
    global total
    total = total + 1 
    print("fetching article", total)
    fp = urlopen(url).read()
    print ("DONE")
    return fp
    

def get_text_from_HTML(html):
    # getting the raw text from all p tags with the class dre-article-body__paragraph')  
    soup = BeautifulSoup(html, features="html.parser")
    result = ""
    paragraphs = soup.findAll('p', attrs={'class':'dre-article-body__paragraph'})
    
    newline = ""
    for p in paragraphs:
        result = result + p.text + newline
        newline = "\n"

    return result


def get_text_from_url(id):
    # print(id)
    data.loc[id, 'Text'] = get_text_from_HTML(getHTML(data.loc[id, 'URL']))
    return 


threads = min(MAX_THREADS, len(data['URL']))
data['HTML'] = [None] * len(data['URL'])
data['Text'] = [None] * len(data['URL'])
with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
     executor.map(get_text_from_url, data.index)
    
print("DONE WITH ALL")

print("WRITE TO FILE")
print('Total time:', time.time()-start_time)
data.to_csv("data/test.csv", encoding="utf-16", sep='\t')

