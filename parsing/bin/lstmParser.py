# -*- coding: utf-8 -*-
import pickle
import re
import sys
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import serializers
from chainer import Variable
import gensim
import numpy as np
from chainer import optimizers

sys.path.append('../lib/')
from loader import myLoader

class Parser(chainer.Chain):
    def __init__(self):
        self.raw_input_dim = 49109
        self.output_dim = 95
        self.action_len = 95
        self.POS_len, self.POS_ex = 45, 20
        self.midOne, self.midTwo = 100, 50
        self.bufDim = 420
        self.stkDim = 700
        self.hisEmbed = 20
        super(Parser, self).__init__(
            #embedWordOfStack = L.EmbedID(self.raw_input_dim, self.midOne),
            embedWordId = L.EmbedID(self.raw_input_dim, self.midOne),
            embedHistoryId = L.EmbedID(self.action_len, self.midOne),
            embedActionId = L.EmbedID(self.action_len, self.midOne),
            embedPOSId = L.EmbedID(self.POS_len, self.POS_ex),
            U = L.Linear(self.stkDim, self.midOne),  # stkInput => lstm
            V = L.Linear(self.bufDim, self.midOne),  # bufInput => lstm
            LS = L.LSTM(self.midOne, self.midTwo),  # for the subtree
            LA = L.LSTM(self.midOne, self.midTwo),  # for the action history
            LB = L.LSTM(self.midOne, self.midTwo),  # for the buffer
            W = L.Linear(self.midTwo*3, self.midTwo*2), # [St;At;Bt] => classifier
            G = L.Linear(self.midTwo*2, self.output_dim)  # output
    )



    def reset_state(self):
        self.LS.reset_state()
        self.LA.reset_state()
        self.LB.reset_state()

    def __call__(self, train, label):
        """
        param: {
                x: {
                    his: historyID INT,
                    buf: {
                        w, WordID INT
                        wlm, pre-trained word2vec np.ndarray(dtype=np.float32)
                        t, POS tag ID INT
                        },
                     stk:{
                         h: HEAD pre-trained word2vec np.ndarray(dtype=np.float32)
                         d: DEPENDENT pre-trained word2vec np.ndarray(dtype=np.float32)
                         r: actionID tag INT
                        }
                }
                y: Labels one-hot-Vector np.ndarray(dtype=np.float32)
            }
        return: softmax_cross_entropy(h3,y) Variable
        """
        his, buf, stk = train[0], train[1], train[2]
        label = Variable(np.asarray(
            [1 if i == label else 0 for i in range(self.action_len)],
            dtype=np.int32))
        label = label.reshape(1,self.action_len)

        his = self.embedHistoryId(np.asarray([his],dtype=np.int32))
        print("his:",his)
        buf = F.concat(
            (self.embedWordId(np.asarray([buf[0]],dtype=np.int32)),
            Variable(buf[1]).reshape(1,300),
            self.embedPOSId(np.asarray([buf[2]],dtype=np.int32))))
        print("buf:",buf)

        stk = F.concat((Variable(stk[0]).reshape(1,300),
                        Variable(stk[1]).reshape(1,300),
                        self.embedActionId(np.asarray([stk[2]],dtype=np.int32))))
        print("stk:",stk)

        # apply U,V
        stk = self.U(stk)
        stk = F.relu(stk)
        buf = self.V(buf)
        buf = F.relu(buf)

        # apply LSTMs
        at = self.LA(his)
        at = F.relu(at)
        st = self.LS(stk)
        st = F.relu(st)
        bt = self.LB(buf)
        bt = F.relu(bt)

        print("---midPoint---")
        # final stage
        h1 = F.concat((st, at, bt))
        h2 = self.W(h1)
        h2 = F.relu(h2)
        h3 = self.G(h2)
        # pred = F.Softmax(h3)

        return F.softmax_cross_entropy(h3,label)


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
            sentence = loader.gen()  # 学習データが尽きるとIndexErrorを吐く

            accumLoss = Variable()
            for step in sentence:
                train, label = step[0], step[1]
                loss = model(train,label)
                accumLoss += loss
            accumLoss.backward()
            optimizer.update()
            parser.reset_stste()
        except IndexError:
            print("index error")
            break

    serializers.save_hdf5("../model/mymodel.h5", model)
