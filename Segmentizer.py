import pandas as pd
import re
from afinn import Afinn




class Segmentizer:
    def __init__(self):
        pass

    def get_segments(text):
        pattern = "(([a-z]|[%!]|[1-9])\.) *([A-Z|Æ|Ø|Å])"
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
            if len(result) < 400: # skip segments longer than 400 characters
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

    