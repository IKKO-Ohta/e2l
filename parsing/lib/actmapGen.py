import glob
import pickle

mrgs = glob.glob("../auto/Penn_conbined_wsj/*/*mrg")

actions = []
for mrg in mrgs:
    with open(mrg) as f:
        for line in f:
            line = line.split("\t")
            if len(line) <= 1:
                actions.append(line[0])

s_actions = set(actions)

act_map = {v: i for i, v in zip(range(len(s_actions)), s_actions)}
with open("../model/act_map.pkl", "wb") as f:
    pickle.dump(act_map, f)
