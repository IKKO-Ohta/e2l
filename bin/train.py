import glob
import sys
import datetime as dt
from copy import copy
import numpy as np
import pandas as pd
sys.path.append("../lib")
from load_article import load_article
from load_tsv import load_tsv

def build(start,end):
    x,y = [],[]
    aDay = copy(start)
    while(1):
        if aDay == END:
            break
        strDate = aDay.strftime("%Y%m%d")
        value = load_tsv(strDate,feature=True)
        article = load_article(strDate)
        if article != int and type(value) != int:
            x.append(value)
            y.append(article)
        aDay = aDay + dt.timedelta(days=1)

    return x,y

if __name__ == '__main__':
    START = dt.date(2012,7,2)
    END   = dt.date(2014,12,31)


    X,Y = build(START,END)

    assert(len(articles) == len(dj))
