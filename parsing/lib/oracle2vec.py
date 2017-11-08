import copy
import sys
import re
import glob
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
        # print([conf.stack[1], relation, conf.stack[0]])
        conf.stack.pop(0)

    @staticmethod
    def left_arc(relation, conf):
        conf.arcs.append([conf.stack[0], relation, conf.stack[1]])
        # print([conf.stack[0], relation, conf.stack[1]])
        conf.stack.pop(1)

    @staticmethod
    def shift(conf):
        idx_wi = conf.buffer.pop(0)
        conf.stack.insert(0, idx_wi)
        if not conf.stack[-1]: del conf.stack[-1]

class myVectorizer(object):
    """
    - cal buffer
    - cal stack
    - 変数のembed
    - ダミー化
    """

    def __init__(self):
        self.regex = re.compile('[a-zA-Z0-9]+')
        #self.wv_model = gensim.models.KeyedVectors.load_word2vec_format('../model/GoogleNews-vectors-negative300.bin',
        #binary=True)
        with open("../model/tag_map.pkl", "rb") as f:
            self.tag_map = pickle.load(f)
        with open("../model/word2id.pkl", "rb") as f:
            self.corpus = pickle.load(f)
        with open("../model/simple_act_map.pkl", "rb") as f:
            self.act_map = pickle.load(f)
        with open("../model/tag2id.pkl", "rb") as f:
            self.tag2id = pickle.load(f)

    def reg(self, word):
        g = self.regex.match(word)
        if g:
            return g.group()
        else:
            return ""

    def find_tag(self, word):
        word = self.regex.match(word)
        return self.tag_map[word]

    def buf_embed(self, buffer):
        if not buffer:
            w = -1
            wlm = -1
            tag = -1
            return [w,wlm,tag]

        word = buffer[0]  # last

        def find(key):
            return self.corpus[key], self.corpus[key], self.tag_map[key]

        def dm_find(key):
            """
            dummy find.
            もしコーパスに探すべきものが見つからなかったら、
            とりあえず-1を返しておく
            """
            return -1, -1, -1

        def not_null(key):
            dicts = [self.corpus, self.tag_map]
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
            w, wlm, tag = dm_find(word)

        print("before:",tag)
        #w = self.dummy(w, len(self.corpus))
        tag = self.tag2id[tag]
        print("after:",tag)
        return [w, wlm, tag]

    def edge_embed(self, head, arcs):
        """
        param: conf.head, conf.arcs
        return: [[h,d,r],[h,d,r]...] # => 全ての要素がID化されている
        """

        def dfs(h,arcs,result):
            """
            arcsと、ルートになる単語が与えられたとき、
            木を構成するエッジを探索して返す再帰関数
            """
            # 停止条件
            if not h in [arc[0] for arc in arcs]:
                return
            # あるheadについて、arcsを全走査
            for arc in arcs:
                if arc[0] == h:
                    result.append(arc)
                    dfs(arc[2],arcs,result)
            return result

        tree = []
        raw_edges = dfs(head,arcs,[])

        # 返すべき木はない
        if not raw_edges:
            h,d,r = -1, -1, -1
            return [h,d,r]

        #　返すべき木がある
        for raw_edge in raw_edges:
            if "LEFT" in raw_edge[1]:
                act = "LEFT"
            elif "RIGHT" in raw_edge[2]:
                act = "RIGHT"
            else:
                act = "SHIFT"
            h = self.reg(raw_edge[0])
            h = self.corpus[h]
            r = self.act_map[act]
            d = self.reg(raw_edge[2])
            d = self.corpus[d]
            tree.append([h,d,r])
        return tree

    def cal_history(self, history):
        """
        :param バッファの最新ヒストリ
        :return: ダミー化されたヒストリ
        """
        if history:
            return self.act_map[history[-1]]
        else:
            return 0


def oracle_load(oracle_path):
    """
    オラクルファイルを受け取って、confに代入できるよう整形する
    featureとラベルが返る
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
            elif line == "\n": # EOS
                break
            else:
                line = line.rstrip()
                label.append(line)
                cnt += 1
    return feature, label

if __name__ == '__main__':
    if len(sys.argv) == 1:
        pathes = glob.glob("../auto/Penn_Oracle_split/*/*.oracle")
    else:
        pathes = glob.glob(sys.argv[1])
    vectorizer = myVectorizer()
    error = 0

    for path in pathes:
        words, actions = oracle_load(path)
        conf = Configuration()
        conf.stack = copy.deepcopy(words[0][0])
        conf.buffer = copy.deepcopy(words[0][1])
        cnt = 0
        try:
            for action in actions:

                his = vectorizer.cal_history(conf.history)
                buf = vectorizer.buf_embed(conf.buffer)
                stk = vectorizer.edge_embed(conf.stack[-1], conf.arcs)


                if action == "SHIFT":
                    Transition.shift(conf)
                elif "RIGHT" in action:
                    Transition.right_arc(action, conf)
                    action = "RIGHT"
                elif "LEFT" in action:
                    Transition.left_arc(action, conf)
                    action = "LEFT"
                else:
                    """
                    予期しない命令
                    """
                    break
                target_dir_num = path.split("/")[-2]
                origin_name = path.split("/")[-1].replace(".oracle", "")
                target_path = "../auto/preprocessed/" + target_dir_num \
                              + "/" + origin_name + "_" + '{0:07d}'.format(cnt) + ".pkl"

                with open(target_path, "wb") as target:
                    #print("gen:", target_path)
                    label = vectorizer.act_map[action]
                    print("his,buf,stk:", [his,buf,stk])
                    print("label:",label)
                    pickle.dump(([his, buf, stk], label), target)

                conf.history.append(action)
                cnt += 1
                if cnt > 100:
                    import pdb; pdb.set_trace()

        except IndexError:
            print("--ERROR-- ", path)
            print("IndexError")
            print("continue..")
            error += 1

    print("preprocess done. found Error ..")
    print(error)
