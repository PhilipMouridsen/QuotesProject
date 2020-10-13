import time
import pandas as pd
import re

print("reading data")
data = pd.read_csv("test100.csv", encoding="utf-16", sep='\t', index_col=0)
print("DONE reading data")

def get_quotes(id):
    text = data.loc[id, "Text"]
    pattern = "((?<=\n- ).*(?=\n))"
    return re.findall(pattern, text)




print(data.info())
print(data.index[0])
data['Quotes'] = data.apply(get_quotes(x), data.index) 

print(data)

print (get_quotes(data.index[2]))






# ((?<=\n- ).*(?=\n))