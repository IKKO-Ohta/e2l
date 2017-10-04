"""
実際には語と素性の対応は辞書として準備しておき、
パーザからは読み出すだけにしておく
"""

import re
import gensim
import pickle


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
        with open("../model/tag_map.pkl", "br") as f:
            self.tag_map = pickle.load(f)
        with open("../model/word2id.pkl", "rb") as f:
            self.corpus = pickle.load(f)
        with open("../model/action2id.pkl", "rb") as f:
            self.act_map = pickle.load(f)

    def reg(self, word):
        g = self.regex.match(word)
        return g.group()

    def find_tag(self, word):
        word = self.regex.match(word)
        return self.tag_map[word]

    def embed(self, word_list):
        word_list = [self.reg(word) for word in word_list]
        return [(self.corpus[word], self.wv_model[word], self.tag_map[word]) for word in word_list]

    def cal_history(self, history):
        return [self.act_map(his) for his in history]



if __name__ == '__main__':
    buffer = ['Motor-_', 'maker-_', 'auto-_', 'Japanese-_', 'of-_',
              'sales-_', 'of-_', 'president-_', 'named-_', 'was-_']

    vec = myVectorizer()
    ans = vec.embed(buffer)
    print(ans)