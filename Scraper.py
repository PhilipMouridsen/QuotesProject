# scraper
import pandas as pd
import urllib.request

data = pd.read_csv("artikelurls10.csv", encoding="utf-16", sep='\t')




print(data.describe)
print(data.head)

print(data.columns)
data = data.drop(columns=['URN, Pub., Format and 2 more (Combined)','Brugere', 'Gns. tid', 'Afs. video', '% Afs. video','SoMe', '% SoMe', 'Search', '% Search', 'URN Count d' ])
print(data)


def getHTML(url):
    fp = urllib.request.urlopen(url)
    bytes = fp.read()
    string = bytes.decode("utf8")
    fp.close()

    return string



data['HTML'] = [getHTML(x) for x in data['URL']]


print (data)