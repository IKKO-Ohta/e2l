import sys
from chainer import serializers
from lstmParser import evaluate
sys.path.append("../lib")
from loader import myloader

print("loading...")
model = serializers.load_hdf5(sys.args[1])
loader = myLoader()
loader.set()

print("evaluate...")
evaluate(model,loader)
