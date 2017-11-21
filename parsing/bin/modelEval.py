import sys
import numpy as np
from chainer import serializers
from lstmParser import evaluate,Parser
sys.path.append("../lib")
from loader import myLoader

print("loading...")
model = Parser()
serializers.load_hdf5(sys.argv[1],model)
loader = myLoader()
loader.set()

print("evaluate...")
pred,gold = evaluate(model,loader)

modelName = sys.argv[1].split("/")[-1].replace(".mod","")

print("print results..")
np.save(pred,"../result/"+modelName+"Pred.npy")
np.save(gold,"../result/"+modelName+"Gold.npy")
