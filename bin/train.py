import glob
import sys
import datetime as dt
import numpy as np
import pandas as pd
from copy import copy
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
