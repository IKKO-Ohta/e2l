import glob
import sys
import datetime as dt
from copy import copy
import numpy as np
import pandas as pd
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
        y_i = load_article(strDate)
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
