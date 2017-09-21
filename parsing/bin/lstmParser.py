import glob
import pickle
import os
from operator import itemgetter
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import report, training, Chain, datasets, iterators, optimizers


# from chainer.training import extensions
# from chainer.datasets import tuple_dataset
# import matplotlib.pyplot as plt


# from chainer.training import extensions

class Configuration(object):
    def __init__(self):
        self.stack = [0]  # The root element
        self.buffer = []
        self.arcs = []  # empty set of arc

    def _build(self, sequence):
        self.buffer = [i for i in range(1, len(sequence))]


class Transition(object):
    def __init__(self):
        return

    @staticmethod
    def _right(relation, conf):
        idx_wj = conf.stack.pop()
        idy_yj = conf.buffer[0]
        conf.arcs.append((idx_wj, relation, idy_yj))

    @staticmethod
    def _left(relation, conf):
        idx_wi = conf.stack[len(conf.stack) - 1]
        conf.stack.pop()
        idx_wj = conf.buffer[0]
        conf.arc.append([idx_wj, relation, idx_wi])

    @staticmethod
    def _shift(conf):
        idx_wi = conf.buffer.pop()
        conf.stack.append(idx_wi)


class Parser(chainer.Chain):
    def __init__(self):
        self.conf = Configuration()
        self.A = []
        super(Parser, self).__init__()
        with self.init_scope():
            """
            TODO: 具体的な次元数を求める => 素性側が出そろったら
            """
            self.embed = L.EmbedID(1000, 100)  # word embedding
            self.LS = L.LSTM(100, 50)  # for the subtree
            self.LA = L.LSTM(100, 50)  # for the action history
            self.LB = L.LSTM(100, 50)  # for the buffer
            self.U = L.Linear(50, 1000)  # input => lstm
            self.V = L.Linear(50, 1000)  # input => lstm
            self.W = L.Linear(50, 1000)  # [St;At;Bt] => classifier
            self.G = L.Linear(50, 1000)  # output

    def _set_parser_resource(self, file_path):
        self.conf.buffer = self.load_conll(file_path)

    def reset_state(self):
        self.LS.reset_state()
        self.LA.reset_state()
        self.LB.reset_state()

    @staticmethod
    def _oracle_dump(dir_path):
        def embed(wordlist):
            div = wordlist.find("][")
            stack = wordlist[:div+1]
            buffer = wordlist[div+1:]
            return stack, buffer
        oracles = glob.glob(dir_path)
        feature, label = [], []
        for oracle in oracles:
            with open(oracle, "r") as f:
                """
                以下はoracle形式のつじつま合わせ
                """
                f.read()
                cnt = 0
                for line in f:
                    if cnt % 2 == 0:
                        feature.append(line)
                        cnt += 1
                    else:
                        label.append(line)
                        cnt += 1
        # 再構築
        dataset = []
        action = []
        for feat, l in zip(feature, label):
            if not feat:
                action = []
            elem = [embed(feat), action, l]
            dataset.append(elem)
            action.append(l)

        # 後でnumpy or pd処理を噛ませるかも
        with open("oracle_dataset.pkl","wb") as f:
            pickle.dump(dataset,f)
        return dataset

    @staticmethod
    def load_conll(file_path):
        if os.path.isfile(file_path):
            loadfiles = [file_path]
        else:
            loadfiles = glob.glob(file_path)
        features = []
        for loadfile in loadfiles:
            with open(loadfile, "r") as f:
                feature = [itemgetter(1, 3)(line.split("\t")) for line in f]
                feature.append(np.ndarray([0 for i in range(100)]))
                features.append(feature)
        return features

    @staticmethod
    def _cal_stack(stack):
        """
        部分木の計算方法は後で調べる
        線形変換Uとreluを再帰的に適用するのは知っている
        """
        return sum(stack)

    def _cal_buffer(self, buffer):
        res = []
        for b in reversed(buffer):
            b1 = self.V2(b)
            b1 = F.relu(b1)
            b1 = self.LB(b1)
            res.append(b1)
        return res[-1]

    def __call__(self):
        # Given the current word ID, predict the next word.
        self.reset_state()
        at = [self.LA(a) for a in reversed(self.A)][0]
        st = self._cal_stack(self.conf.stack)
        bt = self._cal_buffer(self.conf.buffer)
        h1 = np.concatenate([st, at, bt])
        h2 = self.W(h1)
        h2 = F.relu(h2)
        h3 = self.G(h2)
        return F.Softmax(h3)


parser = Parser()
model = L.Classifier(parser)
optimizer = optimizers.SGD()
optimizer.setup(model)
