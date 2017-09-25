import chainer
import chainer.functions as F
import chainer.links as L


class RecNN(chainer.Chain):
    def __init__(self):
        super(RecNN, self).__init__()
        with self.init_scope():
            self.l = L.Linear(50, 1000)

    def __call__(self, a, b, i):
        if len(b) == 1:
            h1 = self.l(a[0], b[0])
            h1 = F.relu(h1)
            return h1
        ans = self.__call__(b[i], b[i + 1:], i + 1)
        return ans
