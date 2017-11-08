import sys
import glob
import pickle

oracles = glob.glob("../../auto/Penn_Oracle_split/*/*.oracle")

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

word2id, cnt = {}, 0
for oracle in oracles:
    features, _ = oracle_load(oracle)
    features = [feature[0] for feature in features]
    for feature in features:
        for elem in feature:
            if not elem in word2id.keys():
                word2id[elem] = cnt
                cnt += 1

with open("../../model/word2id.pkl","wb") as f:
    pickle.dump(word2id,f)
