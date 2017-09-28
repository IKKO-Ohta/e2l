"""
実際には語と素性の対応は辞書として準備しておき、
パーザからは読み出すだけにしておく
"""

import re
import gensim
import pickle

regex = re.compile('[a-zA-Z0-9]+')


def reg(word):
    global regex
    g = regex.match(word)
    return g.group()


def find_tag(word, m):
    return m[word]


def cal_buffer(word, wv, map, corpus):
    word = reg(word)
    return [corpus[word], wv[word], map[word]]


if __name__ == '__main__':
    buffer = ['Motor-_', 'maker-_', 'auto-_', 'Japanese-_', 'of-_',
              'sales-_', 'of-_', 'president-_', 'named-_', 'was-_']

    wv_model = gensim.models.KeyedVectors.load_word2vec_format('../model/GoogleNews-vectors-negative300.bin', binary=True)

    # 品詞
    with open("../model/tag_map.pkl", "br") as f:
        tag_map = pickle.load(f)

    # FoW
    with open("../auto/Penn_concat/gensim_corpora_dict_cF.pkl", "rb") as f:
        corpus = pickle.load(f)

    for buf in buffer:
        print(cal_buffer(buf, wv_model, tag_map, corpus))
