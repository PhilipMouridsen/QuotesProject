from joblib import dump, load
import pandas as pd 
import numpy as np
from segmentizer import Segmentizer

class QuoteClassifier:
    def __init__(self, modelfile, textfile):
        self.model = load('model/'+ modelfile + '.joblib')
        self.text = open(textfile)
        self.segments = Segmentizer.get_segments(self.text)
        X = pd.DataFrame()
        X['Quotes'] = self.segments
        X['label'] = self.model.predict(X['Quotes'])
        X['proba'] = self.model.predict_proba(X['Quotes'])


    def get_quotes(self):
        return self.X

    def print_result(self):
        ENDC = '\033[0m'
        UNDERLINE = '\033[4m'

    

    def main():
        pass




    if __name__ == "__main__":
        main()
