import pandas as pd
import re
from afinn import Afinn




class Segmentizer:

    def get_segments(text, max_length=500, make_doubles=False):
        first = True
        prev_sent = ""
        pattern = "(([a-z]|[%!]|[1-9])\.) *([A-Z|Æ|Ø|Å])"
        
        text = text.replace(' hr. ', ' hr* ')
        text = text.replace(' Hr. ', ' hr* ')
        text = text.replace(' nr. ', ' nr* ')
        text = text.replace('\n', '')


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

                
                # if True, make quote that includes the previous one
                if make_doubles == True and first == False:
                    print('sent:', sent)
                    print('prev:', prev_sent)
                    result.append(prev_sent + " " + sent)
                
                result.append(sent)
                first = False
                prev_sent = sent


        return result

    def textfile_to_dataframe(filename, make_doubles=False):
        data = pd.read_csv(filename, encoding='utf-8', sep='\n', header=None)
        data.columns=['text']    
        data['Quotes'] = data.text.apply(Segmentizer.get_segments, make_doubles=make_doubles)
        data = data.explode('Quotes').reset_index()
        data = data.drop(columns=['text', 'index'])
        return data


def main():
    
    queen = Segmentizer.textfile_to_dataframe("data/queen2019.txt", make_doubles=False)
    pd.set_option('display.max_colwidth', -1)

    pd.set_option('display.max_rows', None)
    print(queen)

    # data.to_csv('data/wiki-segmentized.csv', sep="\t")


if __name__ == "__main__":
    main()

    