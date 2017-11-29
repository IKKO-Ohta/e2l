"""
train.pyは次のプログラムを起動して、if mainの[START:END]の範囲の素性を
できる限り集めます。
 - lib/build.py
    - lib/load_article.py
    - lib/load_tsv.py
    - lib/pkl/[*].pkl ... word2idの辞書
モデルファイルは別立てで用意し、backword()する予定です。
 - bin/simpleModelWithLSTM.py ... 1層LSTMによる生成(書きかけ)
 - bin/MurakamiLSTM.py ... (Murakami,ACL2017の再実装 ...の予定)
 - bin/RNNG.py ... (Proposal Method ...の予定)
"""
import glob
import sys
import datetime as dt
from copy import copy
import numpy as np
import pandas as pd
from chainer import Variable
import chainer.functions as F
import chainer.links as L
sys.path.append("../lib")
from load_article import load_article
from load_tsv import load_tsv
from build import Builder

def build(start,end,builder):
    x,y = [],[]
    aDay = copy(start)
    while(1):
        if aDay == end:
            break
        x_s = builder.buildSmallX(aDay)
        x_l = builder.buildLargeX(aDay)
        y_i = builder.buildY(aDay)
        if type(x_s) != int and type(x_l) != int and type(y_i) != int:
            x.append([x_s,x_s,x_l,x_l])
            y.append(y_i)
        aDay = aDay + dt.timedelta(days=1)
    return x,y

if __name__ == '__main__':
    START = dt.date(2012,7,2)
    END   = dt.date(2014,12,31)
    builder = Builder()
    X,Y = build(START,END,builder)
    assert(len(X) == len(Y))
    print("----feature---")
    print(X)
    print("---label---")
    print(Y)
    print("---build!---")
