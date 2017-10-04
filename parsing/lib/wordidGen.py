
import glob
import pickle

mrgs = glob.glob("../auto/Penn_conbined_wsj/*/*mrg")
m = {}

for mrg in mrgs:
    with open(mrg) as f:
        for line in f:
            line = line.split("\t")
            if len(line) <= 1:
                continue
            m[line[1]] = line[3]

with open("../model/tag_map.pkl", "wb") as f:
    pickle.dump(m, f)


