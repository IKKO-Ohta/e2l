import copy
import re
import gensim
import pickle
import numpy as np


class Configuration(object):
    def __init__(self):
        self.stack = []
        self.buffer = []
        self.arcs = []
        self.history = []

    def show(self):
        print("stack: ", self.stack)
        print("buffer: ", self.buffer)
        print("arcs: ", self.arcs)


class Transition(object):
    def __init__(self):
        return

    @staticmethod
    def right_arc(relation, conf):
        conf.arcs.append([conf.stack[1], relation, conf.stack[0]])
        print([conf.stack[1], relation, conf.stack[0]])
        conf.stack.pop(0)

    @staticmethod
    def left_arc(relation, conf):
        conf.arcs.append([conf.stack[0], relation, conf.stack[1]])
        print([conf.stack[0], relation, conf.stack[1]])
        conf.stack.pop(1)

    @staticmethod
    def shift(conf):
        idx_wi = conf.buffer.pop(0)
        conf.stack.insert(0, idx_wi)

        if not conf.stack[-1]: del conf.stack[-1]



def _oracle_dump(oracle_path):
    """
    オラクルファイルを受け取って、学習可能な形式にして返す
    - [buffer状況,action状況,stack状況,ラベル]
    なるリスト
    1file-in,1datum-out
    """

    def str_eval(s):
        return s[1:len(s) - 1].split(", ")

    def oracle_split(lists_line):
        div = lists_line.find("][")
        stack = lists_line[:div + 1]
        stack = str_eval(stack)
        buffer = lists_line[div + 1:]
        buffer = str_eval(buffer)
        return [x for x in stack if x != " " or not x], [x for x in buffer if x != " " or not x]

    feature, label = [], []
    with open(oracle_path, "r") as f:
        """
        以下はoracle形式のつじつま合わせ
        """
        f.readline()
        cnt = 0
        for line in f:
            if "][" in line:
                line = line.rstrip()
                line = oracle_split(line)
                feature.append(line)
                cnt += 1
            elif line == "\n":
                """
                EOS
                """
                break
            else:
                line = line.rstrip()
                label.append(line)
                cnt += 1
    return feature, label


class myVectorizer(object):
    """
    - cal buffer
    - cal stack
    - 変数のembed
    - ダミー化
    """

    def __init__(self):
        self.regex = re.compile('[a-zA-Z0-9]+')
        self.wv_model = gensim.models.KeyedVectors.load_word2vec_format('../model/GoogleNews-vectors-negative300.bin',
                                                                        binary=True)
        with open("../model/tag_map.pkl", "rb") as f:
            self.tag_map = pickle.load(f)
        with open("../model/word2id.pkl", "rb") as f:
            self.corpus = pickle.load(f)
        with open("../model/act_map.pkl", "rb") as f:
            self.act_map = pickle.load(f)
        with open("../model/tag2id.pkl", "rb") as f:
            self.tag2id = pickle.load(f)

    def reg(self, word):
        g = self.regex.match(word)
        return g.group()

    def find_tag(self, word):
        word = self.regex.match(word)
        return self.tag_map[word]

    @staticmethod
    def dummy(ind, l):
        m = [0 if i != ind else 1 for i in range(l)]
        return np.asarray(m, dtype=np.float32)

    def buf_embed(self, word):
        def find(key):
            return self.corpus[key], self.wv_model[key], self.tag_map[key]

        def e_find(key):
            return self.corpus[key], np.asarray([0 for i in range(300)]), self.tag_map[key]

        def not_null(key):
            dicts = [self.corpus, self.wv_model, self.tag_map]
            if all([key in d for d in dicts]):
                return True
            else:
                return False

        # 正規表現でwordを洗浄
        word = self.reg(word)

        if not_null(word):
            w, wlm, tag = find(word)
        elif not_null(word.capitalize()):
            w, wlm, tag = find(word.capitalize())
        else:
            w, wlm, tag = e_find(word)

        w = self.dummy(w, len(self.corpus))
        tag = self.dummy(self.tag2id[tag], len(self.tag2id))
        return np.concatenate([w, wlm, tag])

    def edge_embed(self, edge):

        # 正規表現でwordを洗浄
        edge[0], edge[2] = self.reg(edge[0]), self.reg(edge[2])

        if edge[0] in self.wv_model:
            h = self.wv_model[edge[0]]
        elif edge[0].capitalize() in self.wv_model:
            h = self.wv_model[edge[0].capitalize()]
        else:
            h = np.asarray([0 for i in range(300)])

        if edge[2] in self.wv_model:
            d = self.wv_model[edge[2]]
        elif edge[2].capitalize() in self.wv_model:
            d = self.wv_model[edge[2].capitalize()]
        else:
            d = np.asarray([0 for i in range(300)])

        r = self.act_map[edge[1]]
        r = self.dummy(r, len(self.act_map))
        return np.concatenate([h, d, r])

    def cal_history(self, history):
        """
        スタティックメソッドのdummyを用いる
        :param バッファのヒストリ部分
        :return: ダミー化されたhistory
        """
        act = [self.act_map[his] for his in history]
        return [self.dummy(a, len(self.act_map)) for a in act]


if __name__ == '__main__':
    path = "../auto/Penn_Oracle/00/wsj_0009.oracle"
    words, actions = _oracle_dump(path)
    conf = Configuration()
    conf.stack = copy.deepcopy(words[0][0])
    conf.buffer = copy.deepcopy(words[0][1])

    for action in actions:
        if action == "SHIFT":
            Transition.shift(conf)
        elif "RIGHT" in action:
            Transition.right_arc(action, conf)
        elif "LEFT" in action:
            Transition.left_arc(action, conf)
        conf.history.append(action)
