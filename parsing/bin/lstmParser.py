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
import gensim
import numpy as np
from chainer import optimizers

sys.path.append('../lib/')
from loader import myLoader

class Parser(chainer.Chain):
    def __init__(self):
        self.raw_input_dim = 49109
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
                        wlm, pre-trained word2vec np.ndarray(dtype=np.float32)
                        t, POS tag ID INT
                        },
                     stk:{
                         h: HEAD pre-trained word2vec np.ndarray(dtype=np.float32)
                         d: DEPENDENT pre-trained word2vec np.ndarray(dtype=np.float32)
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
        hiss,bufs,stks = 0,0,0
        for train in trains:
            his, buf, stk = train[0], train[1], train[2]
            """
            このあたりのassertはうまく処理を考える
            無くてうまくいくならないほうがいい

            # type assert
            try:
                assert(buf[1].dtype == np.float32)
            except:
                buf[1] = np.asarray(buf[1],dtype=np.float32)
            try:
                assert(stk[0].dtype == stk[1].dtype)
            except:
                stk[0] = np.asarray(stk[0],dtype=np.float32)
                stk[1] = np.asarray(stk[1],dtype=np.float32)
            """
            # his
            his = self.embedHistoryId(np.asarray([his],dtype=np.int32))
            hiss = F.vstack([hiss,his]) if type(hiss) != int else his


            # buf
            if buf == [-1,-1,-1]:
                buf = Variable(np.asarray([0 for i in range(self.bufDim)],dtype=int32))
            else:
                try:
                    embed = self.embedWordPreFix[buf[1]]
                except:
                    embed = np.asarray([0 for i in range(300)],dtype=np.float32)
                try:
                    buf = F.concat(
                        (self.embedWordId(np.asarray([buf[0]],dtype=np.int32)),
                        Variable(embed).reshape(1,300),
                        self.embedPOSId(np.asarray([buf[2]],dtype=np.int32))))
                except:
                    import pdb; pdb.set_trace()
            bufs = F.vstack([bufs,buf]) if type(bufs) != int else buf

            # stk
            compose = 0
            for elem in stk[::-1]:
                if not compose:
                    edge = F.concat(
                    (self.embedWordId(np.asarray([elem[0]],dtype=np.int32)),
                    self.embedWordId(np.asarray([elem[1]],dtype=np.int32)),
                    self.embedActionId(np.asarray([elem[2]],dtype=np.int32))))
                    compose = self.U(edge)
                else:
                    edge = F.concat((
                        compose,
                        self.embedWordId(np.asarray([elem[1]],dtype=np.int32)),
                        self.embedActionId(np.asarray([elem[2]],dtype=np.int32))
                    ))

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

        # final stage
        h1 = F.concat((st, at, bt))
        h2 = self.W(h1)
        h2 = F.relu(h2)
        h3 = self.G(h2)

        return F.softmax_cross_entropy(h3,label)


    def pred(self, his, buf, stk):
        stk = self.U(stk)
        stk = F.relu(stk)
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

def composeTensor(loader,model,test=False):
    hisTensor,bufTensor,stkTensor,testMat = [],[],[],[]
    while(1):
        try:
            if test == False:
                sentence = loader.gen()
            else:
                sentence = loader.genTestSentence()
        except IndexError:
            print("---loader finished---")
            break

        trains = [sentence[i][0] for i in range(len(sentence))]
        tests = [sentence[i][1] for i in range(len(sentence))]
        hisMat, bufMat, stkMat = model.minibatchTrains(trains)
        testVec = Variable(np.asarray(tests,dtype=np.int32))

        hisTensor.append(hisMat)
        bufTensor.append(bufMat)
        stkTensor.append(stkMat)
        testMat.append(testVec)

    return hisTensor, bufTensor, stkTensor, testMat

def backupModel(model,epoch,dirpath="../model/"):
    now = datetime.datetime.now()
    modelName = "../model/parserModel" +"_"+ "ep" + str(epoch) +"_"+ now.strftime('%s') + ".mod"
    serializers.save_hdf5(modelName, model)
    return

def evaluate(model, loader):
    loader.testMode()
    hisTensor, bufTensor, stkTensor, testMat = composeTensor(loader,model,test=True)
    correct, cnt = 0, 0
    for hisMat, bufMat, stkMat, testVec in zip(hisTensor, bufTensor, stkTensor,testMat):
        predcls = model.pred(hisMat,bufMat,stkMat)
        for pred, test in zip(predcls, testVec):
            if pred.data == test.data:
                 correct += 1
            cnt += 1
        model.reset_state()
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
    print("loading..")
    hisTensor, bufTensor, stkTensor, testMat = composeTensor(loader,model)
    print("loaded!")

    """
    Main LOGIC
    """
    timecnt = 0
    for epoch in range(2):
        epochTimeStart = time.time()
        for hisMat, bufMat, stkMat, testVec in zip(hisTensor, bufTensor, stkTensor,testMat):
            loss = model(hisMat, bufMat, stkMat, testVec)
            loss.backward()
            optimizer.update()
            model.reset_state()

            timecnt += 1
            if timecnt % 100 == 0:
                print("■", end="")

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
