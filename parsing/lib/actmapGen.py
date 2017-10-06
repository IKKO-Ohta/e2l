import glob
import pickle

mrgs = glob.glob("../auto/Penn_Oracle/*/*.oracle")

def my_match(line):
    for key in ["SHIFT", "RIGHT", "LEFT"]:
        if key in line:
            return True
    return False

actions = []
for mrg in mrgs:
    with open(mrg) as f:
        for line in f:
            if not "][" in line and my_match(line):
                line = line.rstrip()
                actions.append(line)

s_actions = set(actions)

act_map = {v: i for i, v in zip(range(len(s_actions)), s_actions)}
with open("../model/act_map.pkl", "wb") as f:
    pickle.dump(act_map, f)
