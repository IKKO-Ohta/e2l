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

        self.input_dim = [49454, 3, 49454]
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

    def __call__(self, s,a,t):
        # Given the current word ID, predict the next word.

        st = self.LS(s)
        st = F.relu(st)
        at = self.LA(a)
        at = F.relu(at)
        bt = self.LB(b)
        bt = F.relu(bt)

        h1 = np.concatenate([st, at, bt])
        h2 = self.W(h1)
        h2 = F.relu(h2)
        h3 = self.G(h2)
        # pred = F.Softmax(h3)
        return F.softmax_cross_entropy(h3,y)


if __name__ == '__main__':
    loader = myLoader()
    loader.set()
    parser = Parser()
    model = L.Classifier(parser)
    optimizer = optimizers.SGD()
    optimizer.setup(model)
    parser.reset_state()
    model.cleargrads()
    with open("../model/act_map.pkl","rb") as f:
        labels = pickle.load(f)

    while(1):
        try:
            sentence = loader.gen()
            for step in sentence:
                x, y = step[0], step[1]
                s, a, b = x[0], x[1], x[2]
                y = labels[y]
                loss = model(x,y)
                loss.backward()
                optimizer.update()
            parser.reset_stste()
        except IndexError:
            break
