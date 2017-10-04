import glob
import re
import pickle
import os
import gensim
from operator import itemgetter
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import report, training, Chain, datasets, iterators, optimizers


class Configuration(object):
    def __init__(self):
        self.stack = []  # The root element
        self.buffer = []
        self.arcs = []  # empty set of arc

    def show(self):
        print("stack: ", self.stack)
        print("buffer: ", self.buffer)
        print("arcs: ", self.arcs)


class Transition(object):
    def __init__(self):
        return

    @staticmethod
    def right_arc(relation, conf):
        conf.arcs.append([conf.stack[1], relation, conf.stack[0]])
        conf.stack.pop(0)

    @staticmethod
    def left_arc(relation, conf):
        conf.arcs.append([conf.stack[0], relation, conf.stack[1]])
        conf.stack.pop(1)

    @staticmethod
    def shift(conf):
        idx_wi = conf.buffer.pop(0)
        conf.stack.insert(0, idx_wi)
        if not conf.stack[-1]:
            del conf.stack[-1]


class RedNN(chainer.Chain):
    def __init__(self):
        self.wv_model = gensim.models.KeyedVectors.load_word2vec_format('../model/GoogleNews-vectors-negative300.bin',
                                                                        binary=True)
        with open("../model/tag_map.pkl", "br") as f:
            self.tag_map = pickle.load(f)
        with open("word2id.pkl", "rb") as f:
            self.corpus = pickle.load(f)

        super(RedNN, self).__init__()
        with self.init_scope():
            self.U = L.Linear(50, 1000)

    def __call__(self, words):
        while(1):
            if len(words) == 1:
                return words[0]
            ret0, ret1 = words[-1], words[-2]
            words.pop()
            words.pop()
            if type(ret0) == str:
                ret0 = self.wv_model[ret0]
            ret = np.concatenate([ret0, self.wv_model[ret1], self.tag_map[ret0]])
            ret = self.U(ret)
            ret = F.tanh(ret)
            words.append(ret)

class CalBuffer(object):
    def __init__(self):
        self.wv_model = gensim.models.KeyedVectors.load_word2vec_format('../model/GoogleNews-vectors-negative300.bin',
                                                                        binary=True)
        with open("../model/tag_map.pkl", "br") as f:
            self.tag_map = pickle.load(f)
        with open("word2id.pkl", "rb") as f:
            self.corpus = pickle.load(f)
        self.regex = re.compile('[a-zA-Z0-9]+')
        return

    def reg(self, word):
        g = self.regex.match(word)
        return g.group()

    def find_tag(self, word):
        return self.tag_map[word]

    def __call__(self, word):
        word = self.reg(word)
        return np.concatenate([self.corpus[word], self.wv_model[word], self.tag_map[word]])



class CalStack(object):
    """
    cal stackの手続き
    1. 部分木を作成する
    2. 1つあたりの部分木を計算して返却する
    結局、cal_stack内部で何が起きているのか、我々は知らないのが問題
    オラクルエッジをそのまま受け取れば良い
    []
    入力:  stackとconf.arc
    出力： [部分木1当たりの]ベクトル
    """
    def __init__(self):
        self.red = RedNN()
        self.regex = re.compile('[a-zA-Z0-9]+')

    def reg(self, word):
        g = self.regex.match(word)
        return g.group()

    def reconstruct(self, tree):
        """
        edgesを受け取り、
        [A,B,C,D ... ]のかたちにする
        ついでにA,B,C,Dを整形する
        """
        ans = [edge[0] for edge in tree]
        ans.append(tree[-1][-1])  # head-word
        return [self.reg(a) for a in ans]

    def __call__(self, tree):
        tree = self.reconstruct(tree)
        return self.red(tree)


class Parser(chainer.Chain):
    def __init__(self):
        self.conf = Configuration()
        self.A = []
        self.cal_buffer = CalBuffer()
        self.cal_stack = CalStack()
        super(Parser, self).__init__()
        with self.init_scope():
            """
            TODO: 具体的な次元数を求める => 素性側が出そろったら
            """
            self.embed = L.EmbedID(1000, 100)  # word embedding
            self.LS = L.LSTM(100, 50)  # for the subtree
            self.LA = L.LSTM(100, 50)  # for the action history
            self.LB = L.LSTM(100, 50)  # for the buffer
            self.U = L.Linear(50, 1000)  # input => lstm
            self.V = L.Linear(50, 1000)  # input => lstm
            self.W = L.Linear(50, 1000)  # [St;At;Bt] => classifier
            self.G = L.Linear(50, 1000)  # output

    def reset_state(self):
        self.LS.reset_state()
        self.LA.reset_state()
        self.LB.reset_state()

    def __call__(self):
        # Given the current word ID, predict the next word.
        self.reset_state()
        at = [self.LA(a) for a in reversed(self.A)][0]
        """
        stとbtはループになっているはず
        """
        for b in self.conf.buffer:
            bt = self.cal_buffer(b)
            bt = self.V(bt)
            bt = F.relu(bt)
        for s in self.conf.stack:
            st = self.cal_stack(s)
            st = self.U(st)
            st = F.relu(st)
        h1 = np.concatenate([st, at, bt])
        h2 = self.W(h1)
        h2 = F.relu(h2)
        h3 = self.G(h2)
        return F.Softmax(h3)

    def _creating_traning_example(self, oracles):


        return


if __name__ == '__main__':
    parser = Parser()
    model = L.Classifier(parser)
    optimizer = optimizers.SGD()
    optimizer.setup(model)

    parser.reset_state()
    model.cleargrads()

    loss = compute_loss(x_list,y_list)
    loss.backward()
    optimizer.update()