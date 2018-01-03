import glob
import pickle

def act2idGen(mrgs):
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

    act_map = {v: i for i, v in zip(range(1,len(s_actions)+1), s_actions)}
    act_map["empty"] = 0
    with open("../model/act_map.pkl", "wb") as f:
        pickle.dump(act_map, f)

if __name__ == '__main__':
    mrgs = glob.glob("../auto/Penn_Oracle/*/*.oracle")
    act2idGen(mrgs)
