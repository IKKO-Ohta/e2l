# -*- coding: utf-8 -*-                                                                                                   

"""
以前に作ったbs4による読み込みスクリプト。現状だと見出しだけを拍。
これを改善して一日あたりの概況記事の集合をファイル群として作ろうと思うが、
あまりにも効率が悪くつらい気持ちとなる。
"""
from bs4 import  BeautifulSoup
import re
import numpy as np
import glob
import codecs
from collections import OrderedDict
from sklearn.feature_extraction.text import CountVectorizer


def preprocess(text):
    text = re.sub('\n', '',text)
    text = re.sub('    ','\n',text)
    return text

files_path = glob.glob("../resource/corpus/Dowjones200407_201506/201207-201306/*.nml")

for file_path in files_path:
    with open(file_path,'r') as f:
        print("loading..",file_path)
        soup = BeautifulSoup(f.read(), 'html.parser')
        print("loaded")
        for t in soup.find_all('doc'):
            print(t.find('djnml').get('docdate'))
            if 'SNAPSHOT' in t.headline.string:
                headline = preprocess(t.find("headline").get_text())
                text = preprocess(t.find("text").get_text())
                day = t.find('djnml').get('docdate')
                with open("../auto/djnml_daily_headline/"+day+".txt","w") as f:
                    f.write(headline)
                    f.write("\n")
                    f.write(text)
            else:
                continue
        print("DONE")
