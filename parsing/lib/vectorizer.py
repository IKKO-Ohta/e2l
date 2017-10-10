"""
実際には語と素性の対応は辞書として準備しておき、
パーザからは読み出すだけにしておく
"""

import re
import gensim
import pickle
import numpy as np

class myVectorizer(object):
    """
    - cal buffer
    - cal stack
    - 変数のembed
    - ダミー化
    """

    def __init__(self):
        self.regex = re.compile('[a-zA-Z0-9]+')
        self.wv_model = gensim.models.KeyedVectors.load_word2vec_format('../model/GoogleNews-vectors-negative300.bin',
                                                                        binary=True)
        with open("../model/tag_map.pkl", "rb") as f:
            self.tag_map = pickle.load(f)
        with open("../model/word2id.pkl", "rb") as f:
            self.corpus = pickle.load(f)
        with open("../model/simple_act_map.pkl", "rb") as f:
            self.act_map = pickle.load(f)
        with open("../model/tag2id.pkl", "rb") as f:
            self.tag2id = pickle.load(f)

    def reg(self, word):
        g = self.regex.match(word)
        return g.group()

    def find_tag(self, word):
        word = self.regex.match(word)
        return self.tag_map[word]

    @staticmethod
    def dummy(ind, l):
        m = [0 if i != ind else 1 for i in range(l)]
        return np.asarray(m, dtype=np.float32)

    def embed(self, word_list):
        def find(key):
            return self.corpus[key], self.wv_model[key], self.tag_map[key]
        
        def e_find(key):
            return self.corpus[key], np.asarray([0 for i in range(300)]), self.tag_map[key]
        
        def not_null(key):
            dicts = [self.corpus,self.wv_model,self.tag_map]
            if all([key in d for d in dicts]):
                return True
            else:
                return False
        
        word_list = [self.reg(word) for word in word_list]
        result = []
        for word in word_list:
            if not_null(word):
                w, wlm, tag = find(word)
            elif not_null(word.capitalize()):
                w, wlm, tag = find(word.capitalize())
            else:
                w, wlm, tag = e_find(word)
            
            w = self.dummy(w,len(self.corpus))
            tag = self.dummy(self.tag2id[tag],len(self.tag2id))
            result.append(np.concatenate([w, wlm, tag]))
        return result

    def cal_history(self, history):
        """
        スタティックメソッドのdummyを用いる
        :param バッファのヒストリ部分
        :return: ダミー化されたhistory
        """
        act = [self.act_map(his) for his in history]
        return [self.dummy(a, len(self.act_map)) for a in act]


if __name__ == '__main__':
    buffer = ['Motor-_', 'maker-_', 'auto-_', 'Japanese-_', 'of-_',
              'sales-_', 'of-_', 'president-_', 'named-_', 'was-_']
    history = [10, 20, 30, 11]
    vec = myVectorizer()
    buf = vec.embed(buffer)
    his = vec.cal_history(history)
    print(his)
