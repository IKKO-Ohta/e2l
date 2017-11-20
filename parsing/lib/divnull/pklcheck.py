import pickle
import glob

pkls = glob.glob("../../auto/preprocessed/13/*.pkl")
for pkl in pkls:
    with open(pkl,"br") as f:
        pkl = pickle.load(f)
        pkl = pkl[0]
        his,buf,stk = pkl[0],pkl[1],pkl[2]
        if not 0 <= his <= 2:
            print("found..")
            print(pkl)
            print(his,buf,stk)
