import glob
import pickle
import os
import gensim
from operator import itemgetter
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import report, training, Chain, datasets, iterators, optimizers


# from chainer.training import extensions
# from chainer.datasets import tuple_dataset
# import matplotlib.pyplot as plt


# from chainer.training import extensions

def make_vocab():
    return


class Configuration(object):
    def __init__(self):
        self.stack = [0]  # The root element
        self.buffer = []
        self.arcs = []  # empty set of arc

    def _build(self, sequence):
        self.buffer = [i for i in range(1, len(sequence))]


class Transition(object):
    def __init__(self):
        return

    @staticmethod
    def _right(relation, conf):
        idx_wj = conf.stack.pop()
        idy_yj = conf.buffer[0]
        conf.arcs.append((idx_wj, relation, idy_yj))

    @staticmethod
    def _left(relation, conf):
        idx_wi = conf.stack[len(conf.stack) - 1]
        conf.stack.pop()
        idx_wj = conf.buffer[0]
        conf.arc.append([idx_wj, relation, idx_wi])

    @staticmethod
    def _shift(conf):
        idx_wi = conf.buffer.pop()
        conf.stack.append(idx_wi)


class Parser(chainer.Chain):
    def __init__(self):
        self.conf = Configuration()
        self.A = []
        self.wvmodel = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin',
                                                                       binary=True)
        with open("../auto/Penn_concat/gensim_corpora_dict_cF.pkl") as f:
            self.corpora = pickle.load(f)
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

    def _cal_stack(self, e_stack):
        """
        e_stackは部分木が計算されてベクトルになった状態
        cal_bufferと同等の処理を計算する
        """
        res = []
        for s in reversed(e_stack):
            s1 = self.U(s)
            s1 = F.relu(s1)
            s1 = self.LS(s1)
            res.append(s1)
        return res[-1]

    def _cal_buffer(self, e_buffer):
        res = []
        for b in reversed(e_buffer):
            b1 = self.V(b)
            b1 = F.relu(b1)
            b1 = self.LB(b1)
            res.append(b1)
        return res[-1]

    def __call__(self):
        # Given the current word ID, predict the next word.
        self.reset_state()
        at = [self.LA(a) for a in reversed(self.A)][0]
        st = self._cal_stack(self.conf.stack)
        bt = self._cal_buffer(self.conf.buffer)
        h1 = np.concatenate([st, at, bt])
        h2 = self.W(h1)
        h2 = F.relu(h2)
        h3 = self.G(h2)
        return F.Softmax(h3)


parser = Parser()
model = L.Classifier(parser)
optimizer = optimizers.SGD()
optimizer.setup(model)
