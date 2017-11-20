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

sys.path.append('../lib/')
from loader import myLoader

gpu_device = 0
cuda.get_device(gpu_device).use()
model.to_gpu(gpu_device)
xp = cuda.cupy

class Parser(chainer.Chain):
    def __init__(self):
        self.raw_input_dim = 49111
        self.output_dim = 3
        self.action_len = 3
        self.w2vdim = 300
        self.POS_len, self.POS_ex = 45, 20
        self.midOne, self.midTwo = 100, 50
        self.bufDim = self.midOne + 300 + self.POS_ex
        self.stkDim = self.midOne * 2 + self.action_len
        self.embedWordPreFix = gensim.models.KeyedVectors.load_word2vec_format(
            '../model/GoogleNews-vectors-negative300.bin',binary=True)

        super(Parser, self).__init__(
            #embedWordOfStack = L.EmbedID(self.raw_input_dim, self.midOne),
            embedWordId = L.EmbedID(self.raw_input_dim, self.midOne),
            embedHistoryId = L.EmbedID(self.action_len, self.action_len),
            embedActionId = L.EmbedID(self.action_len, self.action_len),
            embedPOSId = L.EmbedID(self.POS_len, self.POS_ex),
            U = L.Linear(self.stkDim, self.midOne),  # stkInput => lstm
            V = L.Linear(self.bufDim, self.midOne),  # bufInput => lstm
            LS = L.LSTM(self.midOne, self.midTwo),  # for the subtree
            LA = L.LSTM(self.action_len, self.action_len),  # for the action history
            LB = L.LSTM(self.midOne, self.midTwo),  # for the buffer
            W = L.Linear(self.midTwo*2 + self.action_len, self.midTwo), # [St;At;Bt] => classifier
            G = L.Linear(self.midTwo, self.output_dim)  # output
    )


    def minibatchTrains(self,trains):
        """
        param:
        trains:{
                x_i: {
                    his: historyID INT,
                    buf: {
                        w, WordID INT
                        wlm, pre-trained word2vec xp.ndarray(dtype=xp.float32)
                        t, POS tag ID INT
                        },
                     stk:{
                         h: HEAD pre-trained word2vec xp.ndarray(dtype=xp.float32)
                         d: DEPENDENT pre-trained word2vec xp.ndarray(dtype=xp.float32)
                         r: actionID tag INT
                        }
                }
                x_i+1:{
                    his: ...,
                    ...,
                }
            }
        return: minibatch his,buf,stk
        """
        errorcnt = 0
        hiss,bufs,stks = 0,0,0
        for train in trains:
            his, buf, stk = train[0], train[1], train[2]

            # his
            his = self.embedHistoryId(xp.asarray([his],dtype=xp.int32))
            hiss = F.vstack([hiss,his]) if type(hiss) != int else his


            # buf
            if buf == [-1,-1,-1]:
                buf = xp.asarray([0 for i in range(self.bufDim)],dtype=xp.float32)
                buf = Variable(buf).reshape(1,self.bufDim)
            else:
                """
                w2vモジュールが悪い例外を吐くため
                """
                try:
                    embed = self.embedWordPreFix[buf[1]]
                except:
                    embed = xp.asarray([0 for i in range(300)],dtype=xp.float32)

                buf = F.concat(
                        (self.embedWordId(xp.asarray([buf[0]],dtype=xp.int32)),
                        Variable(embed).reshape(1,300),
                        self.embedPOSId(xp.asarray([buf[2]],dtype=xp.int32))))
            bufs = F.vstack([bufs,buf]) if type(bufs) != int else buf

            # stk
            compose = 0
            for elem in stk[::-1]:
                """
                elem = -1
                のスカラーで回ってくることがある：
                ../auto/preprocessed/13/wsj_1353_21_0000000.pkl周辺で発生
                """
                if type(compose) == int:
                    try:
                        edge = F.concat(
                        (self.embedWordId(xp.asarray([elem[0]],dtype=xp.int32)),
                        self.embedWordId(xp.asarray([elem[1]],dtype=xp.int32)),
                        self.embedActionId(xp.asarray([elem[2]],dtype=xp.int32))))
                    except:
                        sys.stderr.write("---stk loading error---")
                        sys.stderr.write("--- stk := [[-1,-1,0]]")
                        errorcnt += 1
                        edge = F.concat(
                        (self.embedWordId(xp.asarray([-1],dtype=xp.int32)),
                        self.embedWordId(xp.asarray([-1],dtype=xp.int32)),
                        self.embedActionId(xp.asarray([0],dtype=xp.int32))))

                    compose = self.U(edge)
                    compose = F.relu(compose)
                else:
                    edge = F.concat((
                        compose,
                        self.embedWordId(xp.asarray([elem[1]],dtype=xp.int32)),
                        self.embedActionId(xp.asarray([elem[2]],dtype=xp.int32))
                    ))
                    compose = self.U(edge)
                    compose = F.relu(compose)

            stks = F.vstack([stks, compose]) if type(stks) != int else compose

        return hiss,bufs,stks

    def reset_state(self):
        self.LS.reset_state()
        self.LA.reset_state()
        self.LB.reset_state()

    def __call__(self, his, buf, stk, label):
        """
        params:
            his: {his}, {his}, {...}
            buf: {w,wlm,t}, {...}, {...}
            stk: {h,d,r}, {...}, {...}
            label: y0,y1,y2 ...
        """
        # apply V
        buf = self.V(buf)
        buf = F.relu(buf)

        # apply LSTMs
        at = self.LA(his)
        at = F.relu(at)
        st = self.LS(stk)
        st = F.relu(st)
        bt = self.LB(buf)
        bt = F.relu(bt)

        # final stage
        h1 = F.concat((st, at, bt))
        h2 = self.W(h1)
        h2 = F.relu(h2)
        h3 = self.G(h2)

        return F.softmax_cross_entropy(h3,label)


    def pred(self, his, buf, stk):
        buf = self.V(buf)
        buf = F.relu(buf)
        at = self.LA(his)
        at = F.relu(at)
        st = self.LS(stk)
        st = F.relu(st)
        bt = self.LB(buf)
        bt = F.relu(bt)
        h1 = F.concat((st, at, bt))
        h2 = self.W(h1)
        h2 = F.relu(h2)
        h3 = self.G(h2)
        return F.argmax(h3, axis=1)

def composeMatrix(loader,model,test=False):
    try:
        if test == False:
            sentence = loader.gen()
        else:
            sentence = loader.genTestSentence()
    except IndexError:
        print("---loader finished---")
        return 0

    trains = [sentence[i][0] for i in range(len(sentence))]
    labelVec = [sentence[i][1] for i in range(len(sentence))]
    hisMat, bufMat, stkMat = model.minibatchTrains(trains)
    labelVec = Variable(xp.asarray(labelVec,dtype=xp.int32))

    return [hisMat,bufMat,stkMat,labelVec]

def backupModel(model,epoch,dirpath="../model/"):
    now = datetime.datetime.now()
    modelName = "../model/parserModel" +"_"+ "ep" + str(epoch) +"_"+ now.strftime('%s') + ".mod"
    serializers.save_hdf5(modelName, model)
    return

def evaluate(model, loader):
    correct, cnt = 0, 0
    while(1):
        d = composeMatrix(loader,model,test=True)
        if d:
            hisMat, bufMat, stkMat, testVec = d[0],d[1],d[2],d[3]
            predcls = model.pred(hisMat,bufMat,stkMat)
            for pred, test in zip(predcls, testVec):
                if pred.data == test.data:
                     correct += 1
                cnt += 1
            model.reset_state()
        else:
            break

    print("correct / cnt:", correct, "/", cnt)
    return

if __name__ == '__main__':
    """
    init
    """
    loader = myLoader()
    loader.set()
    model = Parser()
    optimizer = optimizers.SGD()
    optimizer.setup(model)
    model.reset_state()
    model.cleargrads()



    """
    Main LOGIC
    """
    timecnt = 0
    for epoch in range(2):
        epochTimeStart = time.time()
        loader.set()

        while(1):
            d = composeMatrix(loader,model)
            if d:
                hisMat, bufMat, stkMat, testVec = d[0],d[1],d[2],d[3]
                loss = model(hisMat, bufMat, stkMat, testVec)
                loss.backward()
                print("---backward---")
                optimizer.update()
                model.reset_state()
                timecnt += 1
                if timecnt % 100 == 0:
                    print("■", end="")
            else:
                break

        print("epoch",epoch,"finished..")
        print("epoch time:{0}".format(time.time()-epochTimeStart))
        print("backup ..")
        backupModel(model, epoch)
        print("evaluate ..")
        evaluate(model,loader)
        print("Next epoch..")

    print("finish!")
    now = datetime.datetime.now()
    modelName = "../model/parserModel" +"_"+ "complete" +"_"+ now.strftime('%s') + ".mod"
    serializers.save_hdf5(modelName, model)
