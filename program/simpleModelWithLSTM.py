# -*- coding: utf-8-*-
import pickle
import re
import sys
import time
import datetime
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import serializers
from chainer import Variable
from chainer import cuda
import gensim
import numpy as np
from chainer import optimizers

class simpleModelWithLSTM(chainer.Chain):
    def __init__(self):
        self.raw_input_dim = 26885
        self.midOne = 100
        self.midTwo = 50
        super(Parser, self).__init__(
            embedWordId = L.EmbedID(self.raw_input_dim, self.midOne),
            L = L.LSTM(self.midOne, self.midTwo),  # for the subtree
    )
    def __call__(self,x,y):
        """
        param: x,y
        return: loss
        ACL2017を参考にする
        """
        while(1):
            x = self.L(x)
            x = F.relu(x)
            x = self.L1(x)
        # loss = loss(x,y)
        return loss
