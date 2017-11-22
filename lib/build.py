import os
import sys
import pickle
import datetime as dt
import numpy as np
import pandas as pd
from chainer import Variable
import chainer
from load_article import load_article
from load_tsv import load_tsv

class Builder:
    def __init__(self):
        try:
            with open("pkls/word2id.pkl","br") as f:
                self.word2id = pickle.load(f)
            with open("pkls/id2word.pkl","br") as f:
                self.id2word = pickle.load(f)
        except:
            with open("../lib/pkls/word2id.pkl","br") as f:
                self.word2id = pickle.load(f)
            with open("../lib/pkls/id2word.pkl","br") as f:
                self.id2word = pickle.load(f)
        self.dji = pd.read_csv("../auto/dji/DJI2005_2014.csv")

    def buildSmallX(self,date):
        """
        サーベイ通りに実装する
        => まずサーベイをやる、それから実行に移す
        """
        strDate = date.strftime("%Y%m%d")
        df = load_tsv(strDate,feature=True)
        if type(df) != int:
            arr = np.asarray(df).reshape(1,156)
            return Variable(arr)
        else:
            return 0

    def buildLargeX(self,date,windowSize=7):
        result = pd.DataFrame()
        if not date.strftime("%Y-%m-%d") in self.dji["Date"].values:
            return 0
        while(len(result.index) <= windowSize):
            if result.empty:
                result = self.dji.loc[self.dji["Date"] == date.strftime("%Y-%m-%d")]
            else:
                result = pd.concat(
                    [self.dji.loc[self.dji["Date"] == date.strftime("%Y-%m-%d")],result])
            date = date - dt.timedelta(days=1)

        result = np.asarray(result["Close"],dtype=np.int32).reshape(1,windowSize+1)
        return Variable(result)


    def buildY(self,date,headline=True):
        """
        param: date
        return: その日の記事の単語ID列
        現在はheadlineのみを返すようにしている
        """
        sentences = load_article(date.strftime("%Y%m%d"))
        if headline == True and sentences != 0:
            sentence = sentences[0]
            sentence = sentence.replace("\n"," <EOS>")
            sentence = sentence.split(" ")
            Ids = [self.word2id[word] for word in sentence]
            return Variable(np.asarray(Ids,dtype=np.int32))
        elif sentences == 0:
            return 0
        # else..

    def translate(self,Ids):
        """
        param: ID列
        return: 対応する人間用単語列
        """
        Ids = Ids.data
        word = [self.id2word[Id] for Id in Ids]
        return word

if __name__ == '__main__':
    builder = Builder()
    d = dt.date(2014,7,2)
    x_s = builder.buildSmallX(d)
    x_l = builder.buildLargeX(d)
    y = builder.buildY(d)
    print(x_s)
    print(x_l)
    print(y)
    print(builder.translate(y))
