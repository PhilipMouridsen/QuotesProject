import pandas as pd
import re
from afinn import Afinn




class Segmentizer:
    def __init__(self):
        pass

    def get_segments(text):
        pattern = "(([a-z]|[%!]|[1-9])\.) ([A-Z|Æ|Ø|Å])"
        sentences = re.split(pattern, text)

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]


        sentences = chunks(sentences, 4)

        pre = ""
        result = []

        for s in sentences:
            if len(s) > 1:
                sent = pre + s[0] + s[1]
                pre = s[3]
            else: 
                sent = pre+s[0]    
            result.append(sent)

        return result


def main():
    
    data = pd.read_csv("data/wiki.txt", encoding='utf-8', sep='\n')
    data.columns=['text']

    print (data)
    
    senti = Afinn(language='da')

    print(data.iloc[1])

    data['segment'] = data.text.apply(Segmentizer.get_segments)
    data = data.explode('segment').reset_index()
    data = data.drop(columns=['text', 'index'])
    print (data)

    # data.to_csv('data/wiki-segmentized.csv', sep="\t")



if __name__ == "__main__":
    main()

    