# -*- coding: utf-8 -*-
import sys
import pickle
import copy
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import Link, Chain, ChainList, optimizers, Variable, serializers


#  NNの形状を決定する
class Model(Chain):
    def __init__(self):
        super(Model, self).__init__(
            l1=L.Linear(234, 100),
            l2=L.Linear(100, 3064),
        )

    # 誤差関数
    def __call__(self, x, y):
        return F.mean_squared_error(self.fwd(x), y)  # 二乗誤差
    
    # forward
    def fwd(self, x):
        h1 = F.relu(self.l1(x))
        h2 = self.l2(h1)
        return h2

# def get_accuracy(pred, t_test):
#     true_count = 0
#     total_count = 0
#     for pred_one, t in zip(pred.data, t_test.data):
#         total_count+=1
#         pred = np.argmax(pred_one)
#         if pred == t:
#             true_count+=1
#     return true_count/total_count

if __name__ == '__main__':
    # モデルの生成
    model = Model() 

    # optimizerとしてAdamを選択
    optimizer = optimizers.Adam()
    
    # modelにoptimizerをセット
    optimizer.setup(model) 
    
    # 学習データの準備  # xor
    # ndarrayをVariableで囲う
    # 交差検定を行う、devは分割した後方の部分を示す
    # 
    #train_size = 600
    x_train = np.load('./input.npy')
    x_train = Variable(x_train.astype(np.float32))
    #x_train_dev = Variable(np.load('./x_train.npy')[train_size:])
    y_train = np.load('./output.npy')
    y_train = y_train.astype(np.float32)
    y_train = Variable(y_train)
    



    # 学習
    epoch = 50

    developed_accu = 0
#    max_model = 0
    for i in range(1, epoch+1):
        model.zerograds() # 勾配をzeroに
        loss = model(x_train, y_train)  # model.__call__ を呼び出し(誤差計算)
        loss.backward()  # 誤差逆伝播
        optimizer.update()  # パラメータ更新
        print(i, '誤差',loss.data)  # loss -> Variable

        # 以下交差検定を用いる。交差検定の結果がよかったら、MAX_modelを更新する（MAX_modelが最終的に用いられる）
        # pred = model.fwd(x_train_dev)
        # accu = get_accuracy(pred,t_train_dev)
        # if accu > developed_accu:
        #     max_model = copy.deepcopy(model)
        #     developed_accu = accu
        #     print('交差検定による更新')


    # モデルの保存
    serializers.save_npz('my-model', model)
    
    # モデルの保存2
    f = open('my-model.dump', 'wb')
    pickle.dump(model, f)
    f.close()

    # 結果の確認
    x_test = Variable(np.load('./x_test.npy'))
    t_test = pickle.load(open('./t_test.pickle','rb'))
    t_test = Variable(np.asarray(t_test,dtype = np.int32))
    pred = max_model.fwd(x_test)
    print('Accuracy')
    print(get_accuracy(pred,t_test))
