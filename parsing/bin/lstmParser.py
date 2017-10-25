# -*- coding: utf-8-*-
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
        firstIndex = True
        for train in trains:
            his, buf, stk = train[0], train[1], train[2]

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

            # his
            if firstIndex:
                hiss = self.embedHistoryId(np.asarray([his],dtype=np.int32))
            else:
                his = self.embedHistoryId(np.asarray([his],dtype=np.int32))
                hiss = F.vstack([hiss,his])

            # buf
            buf = F.concat(
                (self.embedWordId(np.asarray([buf[0]],dtype=np.int32)),
                Variable(buf[1]).reshape(1,300),
                self.embedPOSId(np.asarray([buf[2]],dtype=np.int32))))
            if firstIndex:
                bufs = buf
            else:
                bufs = F.vstack([bufs,buf])

            # stk
            stk = F.concat((Variable(stk[0]).reshape(1,300),
                            Variable(stk[1]).reshape(1,300),
                            self.embedActionId(np.asarray([stk[2]],dtype=np.int32))))
            if firstIndex:
                stks = stk
            else:
                stks = F.vstack([stks, stk])

            firstIndex = False
        return hiss,bufs,stks

    def reset_state(self):
        self.LS.reset_state()
        self.LA.reset_state()
        self.LB.reset_state()

    def __call__(self, his, buf, stk, label,pred=False):
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
        # pred = F.Softmax(h3)
        if not pred:
            return F.softmax_cross_entropy(h3,label)
        elif pred == True:
            h4 = F.softmax(h3)
            return F.argmax(h4)

def composeTensor(loader,model):
    firstIndex = True
    loader = myLoader()
    loader.set()

    print("Tensor build..")
    while(1):
        sentence = loader.gen()

        try:
            trains = [sentence[i][0] for i in range(len(sentence))]
            tests = [sentence[i][1] for i in range(len(sentence))]
            hisMat, bufMat, stkMat = model.minibatchTrains(trains)
            testVec = Variable(np.asarray(tests,dtype=np.int32))

            if firstIndex == True:
                hisTensor,bufTensor,stkTensor = hisMat,bufMat,stkMat
                testMat = testVec
            else:
                hisTensor = F.vstack([hisTensor,hisMat])
                bufTensor = F.vstack([bufTensor,bufMat])
                stkTensor = F.vstack([sktTensor,stkMat])
                testMat = F.vstack([testMat,testVec])
        except IndexError:
            print("index error")
            break
    print("Loaded ..")
    return hisTensor, bufTensor, stkTensor, testMat

def backupModel(model,epoch,dirpath="../model/"):
    import datetime
    modelName = "parserModel_epoch"+ str(epoch)
                + str(datetime.datetime.now())
    with open(dirpath + modelName,"wb") as f;
        pickle.dump(model,f)
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

    hisTensor, bufTensor, stkTensor, testMat = composeTensor(loader,model)

    """
    Main LOGIC
    """
    for epoch in range(10):
        loss = model(hisTensor, bufTensor, stkTensor, testMat)
        loss.backward()
        optimizer.update()
        model.reset_state()


    for epoch in range(10):
        while(1):
            sentence = loader.gen()
            try:
                trains = [sentence[i][0] for i in range(len(sentence))]
                tests = [sentence[i][1] for i in range(len(sentence))]
                his, buf, stk = model.minibatchTrains(trains)
                tests = Variable(np.asarray(tests,dtype=np.int32))
                print("--forward--")
                loss = model(his, buf, stk, tests)
                print("--backward--")
                loss.backward()
                optimizer.update()
                model.reset_state()
            except IndexError:
                print("index error")
                break
        print("epoch:", epoch, "DONE")
        print("acc..")
        cnt, correct = 0,0
        while(1):
            sentence = loader.genTestSentence()
            gold,pred = [],[]
            try:
                for step in sentence:
                    train,label = step[0],step[1]
                    predCls = model(train,label,pred=True)
                    pred.append(predCls)
                    gold.append(label)
                    if predCls == step[1]:
                        correct += 1
                    cnt += 1
            except:
                print("index Error","Maybe finish")
                break
            print("acc:", correct/cnt)
            print("correct/cnt",correct,"/",cnt)

        print("loader re-initialize"); loader = myLoader()
        print("Next epoch..")


    serializers.save_hdf5("../model/mymodel.h5", model)
