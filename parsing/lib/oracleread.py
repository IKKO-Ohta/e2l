import copy
class Configuration(object):
    def __init__(self):
        self.stack = []  # The root element
        self.buffer = []
        self.arcs = []  # empty set of arc

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
        conf.stack.pop(0)

    @staticmethod
    def left_arc(relation, conf):
        conf.arcs.append([conf.stack[0], relation, conf.stack[1]])
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


if __name__ == '__main__':
    path = "../auto/Penn_Oracle/00/wsj_0009.oracle"
    words, actions = _oracle_dump(path)
    conf = Configuration()
    conf.stack = copy.deepcopy(words[0][0])
    conf.buffer = copy.deepcopy(words[0][1])
    i = 0
    for action in actions:
        if i == 5:
            break
        if action == "SHIFT":
            Transition.shift(conf)
        elif "RIGHT" in action:
            kind = action[action.find("(")+1:action.find(")")]  # "ROOT", "name", ...
            Transition.right_arc(kind, conf)
        elif "LEFT" in action:
            kind = action[action.find("(") + 1:action.find(")")]
            Transition.left_arc(kind, conf)
        i += 1
