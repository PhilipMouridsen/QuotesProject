import time
import pandas as pd
import re
from collections import Counter

print("reading data")
data = pd.read_csv("largefiles/test10000.csv", encoding="utf-16", sep='\t', index_col=0)
print("DONE reading data")

# Get the list of words that ends a typical quote. For example "siger", "udtaler", "uddyber" 
quotewords = []
with open("quotequalifiers.txt", "r", encoding="utf-16") as file:
    quotewords = file.read().splitlines()

# Build a regex of the form "(?:, (?:word1|word2|word3|...)).*" with wordn being a word in the list of quotequalifiers
# This is used to filter out the "atribution-part" of a quote of the form: ", siger statsministeren" or ", udtaler han"
filter_pattern = ""
delimiter = "(?:, (?:"
for word in quotewords:
    filter_pattern = filter_pattern  + delimiter + word
    delimiter = "|"
filter_pattern = filter_pattern + ")).*"



def get_quotes(id):
    text = data.loc[id, "Text"]
    if pd.isna(text): return None
    quote_pattern = r"((?<=\n- ).*(?=\n))" # get everything between "newline-dash-space" and the next "newline" 
    quotes = re.findall(quote_pattern, text)
    if len(quotes) == 0: return None
    for i, quote in enumerate(quotes):
        quotes[i] = re.sub(filter_pattern, "", quote)
        
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

data.to_csv("data/quotes10000.csv", encoding="utf-16", sep='\t')

