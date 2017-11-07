import sys
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
evaluate(model,loader)
