import pickle
import sys
import glob

num = sys.argv[1]
pkls = glob.glob("../../auto/preprocessed/{0}/*.pkl".format(num))
for pklName in pkls:
    with open(pklName,"br") as f:
        pkl = pickle.load(f)
        pkl = pkl[0]
        his,buf,stk = pkl[0],pkl[1],pkl[2]
        if not 0 <= his <= 2:
            print("found..")
            print(pklName)
            print(his)
