import pickle
import glob

pkls = glob.glob("../../auto/preprocessed/12/*.pkl")
for pklName in pkls:
    with open(pklName,"br") as f:
        pkl = pickle.load(f)
        pkl = pkl[0]
        his,buf,stk = pkl[0],pkl[1],pkl[2]
        if not 0 <= his <= 2:
            print("found..")
            print(pklName)
            print(his)
