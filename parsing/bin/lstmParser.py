import pickle
import re
import sys

import chainer
import chainer.functions as F
import chainer.links as L
import gensim
import numpy as np
from chainer import optimizers

sys.path.append('../lib/')
from vectorizer import myVectorizer
from loader import myLoader

class Parser(chainer.Chain):
    def __init__(self):
        self.conf = Configuration()

        self.input_dim = [49454, 3, 49453]
        self.output = 3

        self.vectorizer = myVectorizer()

        super(Parser, self).__init__()
        with self.init_scope():
            """
            TODO: 具体的な次元数を求める => 素性側が出そろったら
            """
            self.LS = L.LSTM(self.input_dim[0], self.input_dim[0])  # for the subtree
            self.LA = L.LSTM(self.input_dim[1], self.input_dim[1])  # for the action history
            self.LB = L.LSTM(self.input_dim[2], self.input_dim[2])  # for the buffer
            self.U = L.Linear(self.input_dim[0], self.input_dim[1])  # input => lstm
            self.V = L.Linear(self.input_dim[0], self.input_dim[1])  # input => lstm
            self.W = L.Linear(sum(self.input_dim), sum(self.input_dim) // 2)  # [St;At;Bt] => classifier
            self.G = L.Linear(sum(self.input_dim) // 2, self.output)  # output

    def reset_state(self):
        self.LS.reset_state()
        self.LA.reset_state()
        self.LB.reset_state()

    def __call__(self):
        # Given the current word ID, predict the next word.
        """

        :: "降ってきたものを食べる”だけ
        ::"なめなおし"はしていない
        """
        self.reset_state()
        at = [self.LA(a) for a in reversed(self.A)][0]
        """
        stとbtはループになっているはず
        """
        self.nullVector = np.asarray([0 for i in range(49454)])

        # 最新のbuffer情報
        if self.conf.buffer:
            bt = self.conf.buffer[-1]
            bt = self.vectorizer.embed(bt)
        else:
            bt = self.nullVector
        bt = self.V(bt)
        bt = F.relu(bt)

        # 最新のエッジ
        if self.conf.stack:
            st = self.conf.arcs[-1]
            st = self.vectorizer.embed(st)
        else:
            st = self.nullVector
        st = self.U(st)
        st = F.relu(st)

        h1 = np.concatenate([st, at, bt])
        h2 = self.W(h1)
        h2 = F.relu(h2)
        h3 = self.G(h2)
        return F.Softmax(h3)


if __name__ == '__main__':
    parser = Parser()
    vectorizer = myVectorizer()
    model = L.Classifier(parser)
    optimizer = optimizers.SGD()
    optimizer.setup(model)

    parser.reset_state()
    model.cleargrads()

    #loss = compute_loss(x_list,y_list)
    ##loss.backward()
    optimizer.update()