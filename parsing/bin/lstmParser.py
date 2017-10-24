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
from vectorizer import myVectorizer
from loader import myLoader

class Parser(chainer.Chain):
    def __init__(self):
        self.raw_input_dim = 49109
        self.output_dim = 95
        self.action_len = 95
        self.POS_len, self.POS_ex = 45, 20
        self.midOne, self.midTwo = 100, 50
        super(Parser, self).__init__(
            #embedWordOfStack = L.EmbedID(self.raw_input_dim, self.midOne),
            embedWordId = L.EmbedID(self.raw_input_dim, self.midOne),
            embedHistoryId = L.EmbedID(self.action_len, self.midOne),
            embedActionId = L.EmbedID(self.action_len, self.midOne),
            embedPOSId = L.EmbedID(self.POS_len, self.POS_ex),
            LS = L.LSTM(self.midOne, self.midTwo),  # for the subtree
            LA = L.LSTM(self.midOne, self.midTwo),  # for the action history
            LB = L.LSTM(self.midOne, self.midTwo),  # for the buffer
            U = L.Linear(self.midOne, self.midTwo),  # input => lstm
            V = L.Linear(self.midOne, self.midTwo),  # input => lstm
            W = L.Linear(self.midTwo*3, self.midTwo*2), # [St;At;Bt] => classifier
            G = L.Linear(self.midTwo*2, self.output_dim)  # output
    )



    def reset_state(self):
        self.LS.reset_state()
        self.LA.reset_state()
        self.LB.reset_state()

    def __call__(self, his,buf,stk):
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

        
        his = self.embedHistoryId(np.asarray([his],dtype=np.int32))
        buf = F.concat((self.embedWordId(np.asarray([buf[0]],dtype=np.int32)),
            buf[1],
            self.embedPOSId(np.asarray([buf[0]],dtype=np.int32))))
        stk = F.concat(
            (stk[0],
            stk[1],
            self.embedActionID(np.asarray([stk[2]],dtype=np.int32))
        ))
        import pdb; pdb.set_trace()
        at = self.LA(his)
        at = F.relu(at)

        st = self.LS(stk)
        st = F.relu(st)

        bt = self.LB(buf)
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
            sentence = loader.gen()  # 学習データが尽きるとIndexErrorを吐く

            accumLoss = Variable()
            for step in sentence:
                x, y = step[0], step[1]
                his, buf, stk = x[0], x[1], x[2]
                y = np.asarray(
                    [1 if i != y else 0 for i in range(len(labels))],
                    dtype=np.float32)
                loss = model(his,buf,stk,y)
                accumLoss += loss
            accumLoss.backward()
            optimizer.update()
            parser.reset_stste()
        except IndexError:
            print("index error")
            break

    serializers.save_hdf5("../model/mymodel.h5", model)
