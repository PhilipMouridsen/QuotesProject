import pandas as pd
import re
from afinn import Afinn




class Segmentizer:

    def get_segments(text, max_length=200):
        pattern = "(([a-z]|[%!]|[1-9])\.) *([A-Z|Æ|Ø|Å])"
        
        text = text.replace(' hr. ', ' hr* ')
        text = text.replace(' Hr. ', ' hr* ')
        text = text.replace(' nr. ', ' nr* ')


        sentences = re.split(pattern,text)

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

                
            if len(sent) < max_length: # skip segments longer than n characters
                sent = sent.replace(' hr* ', ' hr. ')
                sent = sent.replace(' nr* ', ' nr. ')
                sent = sent.replace(' f.eks. ', ' for eksempel ')

                result.append(sent)
        return result

    def textfile_to_dataframe(filename):
        data = pd.read_csv(filename, encoding='utf-8', sep='\n')
        data.columns=['text']    
        data['Quotes'] = data.text.apply(Segmentizer.get_segments)
        data = data.explode('Quotes').reset_index()
        data = data.drop(columns=['text', 'index'])
        return data


def main():
    
    queen = Segmentizer.textfile_to_dataframe("data/queen2019.txt")
    print(queen)

    # data.to_csv('data/wiki-segmentized.csv', sep="\t")


if __name__ == "__main__":
    main()

    