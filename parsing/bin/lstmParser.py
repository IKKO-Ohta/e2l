import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import report, training, Chain, datasets, iterators, optimizers
# from chainer.training import extensions
# from chainer.datasets import tuple_dataset

# import matplotlib.pyplot as plt


# from chainer.training import extensions


class Parser(chainer.Chain):
    def __init__(self):
        self.S = []
        self.A = []
        self.B = []

        super(Parser, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(1000, 100)  # word embedding
            self.LS = L.LSTM(100, 50)  # for the subtree
            self.LA = L.LSTM(100, 50)  # for the action history
            self.LB = L.LSTM(100, 50)  # for the buffer
            self.V1 = L.Linear(50, 1000)  # input => lstm
            self.V2 = L.Linear(50, 1000)  # input => lstm
            self.W = L.Linear(50, 1000)  # [St;At;Bt] => classifier
            self.G = L.Linear(50, 1000)  # output

    def reset_state(self):
        self.LS.reset_state()
        self.LA.reset_state()
        self.LB.reset_state()

    def __call__(self, cur_word):
        # Given the current word ID, predict the next word.

        self.reset_state()
        st = [self.LS(F.ReLU(self.V1(s))) for s in reversed(self.S)][0]
        at = [self.LA(a) for a in reversed(self.A)][0]
        bt = [self.LB(F.ReLU(self.V2(b))) for b in reversed(self.B)][0]
        h1 = np.concatenate([st, at, bt])
        h2 = self.W(h1)
        h2 = F.ReLU(h2)
        h3 = self.G(h2)
        return F.Softmax(h3)


parser = Parser()
model = L.Classifier(parser)
optimizer = optimizers.SGD()
optimizer.setup(model)