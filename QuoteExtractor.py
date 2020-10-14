import time
import pandas as pd
import re
from collections import Counter

print("reading data")
data = pd.read_csv("test100.csv", encoding="utf-16", sep='\t', index_col=0)
print("DONE reading data")

quotewords = []
with open("quotequalifiers.txt", "r", encoding="utf-16") as file:
    quotewords = file.read().splitlines()


def get_quotes(id):
    text = data.loc[id, "Text"]
    if pd.isna(text): return []
    pattern = r"((?<=\n- ).*(?=\n))" # get everything between "newline-dash-space" and the next "newline" 
    quotes = re.findall(pattern, text)
    # r".+?(?=(, (skrev|skriver)))"
    pattern = ""
    delimiter = "(?:, (?:"
    for word in quotewords:
        pattern = pattern  + delimiter + word
        delimiter = "|"
    pattern = pattern + ")).*"
    # print (pattern)
    for i, quote in enumerate(quotes):
        quotes[i] = re.sub(pattern, "", quote)
        
    return quotes

def get_commawords(quote):
    pattern = r"(?<=, )\w*\b"
    words = re.findall(pattern, quote)
    return words

data['Quotes'] = [get_quotes(x) for x in data.index]

# quotelists = data['Quotes']

def commawords_to_file(quotelists, filename):
    commawords = {}
    for quotes in quotelists:
        for q in quotes:
            words = get_commawords(q)
            # print (words)
            for word in words:
                if word not in commawords:
                    commawords[word] = 0
                commawords[word] += 1

    # print (commawords)    
    sorted_list = (Counter(commawords).most_common(len(commawords)))

    with open(filename, 'w', encoding="utf16") as file:
        for key, value in sorted_list:
            file.write("%s, %d\n" % (key, value))

print (get_quotes(data.index[0]))


for id in data.index:    
    # print (data.loc[id, 'Text'])
    print()
    print("QUOTES")
    for quote in data.loc[id, 'Quotes']:
        print(quote)
    print("------------------------")

# commawords_to_file(quotelists, 'test.txt')






# get the word after a comma
# "(?<=, )\w*\b"

# ((?<=\n- ).*(?=\n))