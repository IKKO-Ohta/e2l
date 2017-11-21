import os
import sys
import numpy as np
import pandas as pd
from chainer import Variable
import chaier

class Builder:
    def __init__(self):
        with open("pkls/word2id.pkl","br") as f:
            self.word2id = pickle.load(f)
        with open("pkls/id2word.pkl","br") as f:
            self.id2word = pickle.load(f)

    def buildX(self,df):
        arr = np.asarray(df).reshape(1,156)
        return Variable(arr)

    def buildY(self,sentence):
        Ids = [self.word2id[word] for word in sentence]
        return Variable(np.asarray(Ids,dtype=np.int32))

    def id2word(self,Ids):
        Ids = Ids.data
        word = [self.id2word[Id] for Id in Ids]
        print(" ".join(word))
        return word
