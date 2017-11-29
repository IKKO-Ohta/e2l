# coding: utf-8
#
# NNLM学習（RNN・LSTM・GRU）
#

import sys, os, os.path, time, argparse

# 引数チェック
parser = argparse.ArgumentParser(description='Training NN language model.')
parser.add_argument('mtype',
                    help='Type of model (LSTM/LSTM2/GRU)')
parser.add_argument('dim', type=int,
                    help='Number of EmbedID dimensions')
parser.add_argument('-i', '--iter', type=int, default=15,
                    help='Number of iterations')
parser.add_argument('-b', '--batch', type=int, default=5,
                    help='Batch size')
parser.add_argument('-p', '--par', type=int, default=100,
                    help='Number of parallel batch units per update')
parser.add_argument('-c', '--cutoff', type=int, default=1,
                    help='Word cutoff threshold')
parser.add_argument('-s', '--size', type=int, default=0,
                    help='Training data size (0 to use all)')
parser.add_argument('-n', '--name', default=None,
                    help='Prefix name for output files')
parser.add_argument('-a', '--annotate', action='store_true',
                    help='Create test prob. annotation files (.ann)')
args = parser.parse_args()

mtype = args.mtype.upper()
if mtype != "RNN" and mtype != "LSTM" and mtype != "LSTM2" and mtype != "GRU":
    print("*** Bad model type: %s" % mtype)
    sys.exit(1)

# 分散表現の次元数
demb = args.dim
if demb <= 0:
    print("*** Bad dimension.")
    sys.exit(1)

# 学習の反復回数
num_iter = args.iter
if num_iter < 1:
    print("*** Bad number of iterations.")
    sys.exit(1)

# バッチサイズ
min_batch = args.batch
if min_batch < 1:
    print("*** Bad batch size.")
    sys.exit(1)

# 並列度
par_size = args.par
if par_size < 1:
    print("*** Bad number of parallel batch units.")
    sys.exit(1)

# 単語カットオフ：頻度がこれ以下の単語は未知語扱い
wcutoff = args.cutoff
#if wcutoff < 1:
#    print("*** Bad number of parallel batch units.")
#    sys.exit(1)

# 未知語シンボル
unk = "<unk>"

# 文頭・文末シンボル
#bos = "<s>"
#eos = "</s>"

# 未知語をパープレキシティ計算に含めるか
calc_unk_pp = False

# ディレクトリ
model_base = "model"
train_dir = "train"
test_dir  = "test"

# 実験IDを生成，時刻と合わせて表示
if args.name is None:
    exid = ""
else:
    exid = args.name + "-"

exid += "%s-co%d-d%d-%dx%d" % (mtype, wcutoff, demb, min_batch, par_size)
print("%s" % exid)
print("(%s)\n" % time.asctime())

# 実験用のディレクトリを作成して，モデル等はここに出力
model_dir = model_base + "/" + exid
if os.path.isdir(model_dir) == False:
    os.makedirs(model_dir)

# モジュールの読み込み
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import glob, math

# GPU(CUDA)を使用する
xp = cuda.cupy

# 各モデルのクラス
class MyRNN(Chain):
    def __init__(self, v, k):
        super(MyRNN, self).__init__(
            embed = L.EmbedID(v, k),
            H = L.Linear(k, k),
            W = L.Linear(k, v)
        )
        self.k = k

    # s は numpy (int32) の2次元配列と仮定
    def __call__(self, s):
        accum_loss = None
        h = self.reset()
        for i in range(len(s) - 1):
            h = self.fwd(h, s[i])
            loss = F.softmax_cross_entropy(self.W(h), Variable(s[i + 1]))
            accum_loss = loss if accum_loss is None else accum_loss + loss
        return accum_loss

    def fwd(self, h0, w):
        x = self.embed(Variable(w))
        # ここが引っかかっているので，RNNでは並列処理ができない
        # ( self.H(h0) がブロードキャストされない)
        return F.tanh(x + self.H(h0))

    def reset(self):
        return Variable(xp.zeros((1, self.k), dtype=xp.float32))

class MyLSTM(Chain):
    def __init__(self, v, k):
        super(MyLSTM, self).__init__(
            embed = L.EmbedID(v, k),
            H = L.LSTM(k, k),
            W = L.Linear(k, v)
        )

    # s は numpy (int32) の2次元配列と仮定
    def __call__(self, s):
        accum_loss = None
        h = self.reset()
        for i in range(len(s) - 1):
            h = self.fwd(h, s[i])
            loss = F.softmax_cross_entropy(self.W(h), Variable(s[i + 1]))
            accum_loss = loss if accum_loss is None else accum_loss + loss
        return accum_loss

    # h0 はダミー (実際はHの内部で維持されている)
    def fwd(self, h0, w):
        x = self.embed(Variable(w))
        return self.H(x)

    def reset(self):
        self.H.reset_state()
        return None

class MyLSTM2(Chain):
    def __init__(self, v, k):
        super(MyLSTM2, self).__init__(
            embed = L.EmbedID(v, k),
            H1 = L.LSTM(k, k),
            H2 = L.LSTM(k, k),
            W = L.Linear(k, v)
        )

    # s は numpy (int32) の2次元配列と仮定
    def __call__(self, s):
        accum_loss = None
        h = self.reset()
        for i in range(len(s) - 1):
            h = self.fwd(h, s[i])
            loss = F.softmax_cross_entropy(self.W(h), Variable(s[i + 1]))
            accum_loss = loss if accum_loss is None else accum_loss + loss
        return accum_loss

    # h0 はダミー (実際はHの内部で維持されている)
    def fwd(self, h0, w):
        x = self.embed(Variable(w))
        return self.H2(self.H1(x))

    def reset(self):
        self.H1.reset_state()
        self.H2.reset_state()
        return None

class MyGRU(Chain):
    def __init__(self, v, k):
        super(MyGRU, self).__init__(
            embed = L.EmbedID(v, k),
            H = L.StatefulGRU(k, k),
            W = L.Linear(k, v)
        )

    # s は numpy (int32) の2次元配列と仮定
    def __call__(self, s):
        accum_loss = None
        h = self.reset()
        for i in range(len(s) - 1):
            h = self.fwd(h, s[i])
            loss = F.softmax_cross_entropy(self.W(h), Variable(s[i + 1]))
            accum_loss = loss if accum_loss is None else accum_loss + loss
        return accum_loss

    # h0 はダミー (実際はHの内部で維持されている)
    def fwd(self, h0, w):
        x = self.embed(Variable(w))
        return self.H(x)

    def reset(self):
        self.H.reset_state()
        return None

# インスタンス生成用
def make_model(m):
    if m == "RNN":
        return MyRNN(n_vocab, demb)
    elif m == "LSTM":
        return MyLSTM(n_vocab, demb)
    elif m == "LSTM2":
        return MyLSTM2(n_vocab, demb)
    elif m == "GRU":
        return MyGRU(n_vocab, demb)
    return None


# 学習用の関数
def train_model(model_type, name):
    print("Model type          : %s" % model_type)
    print("Number of embed dims: %d" % demb)
    print("Number of iterations: %d" % num_iter)
    print("Data size to be used: %d" % train_size)
    print("Batch size          : %d" % min_batch)
    print("Parallel batch units: %d" % par_size)
    print("")

    out_base = model_dir + "/" + name + "-e"

    # すでに学習済みのファイルがあるかどうか調べる
    for m in range(num_iter, 0, -1):
        out_file = out_base + ("%02d.model" % m)
        if os.path.exists(out_file):
            break
    else:
        m = 0

    # すでに全epochを学習済みなら何もせず終了
    if m == num_iter:
        print("Found model of epoch %d. No training performed." % m)
        return

    model = make_model(model_type)
    # 学習が途中までなら最後のモデルを読み込み
    if m > 0:
        print("Found model of epoch %d. Restarting training with this model." % m)
        serializers.load_npz(out_file, model)
    # 全くない場合は初期モデル作成
    else:
        model = make_model(model_type)

    cuda.get_device(0).use()
    model.to_gpu()
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    for e in range(m + 1, num_iter + 1):
        print("Epoch %d (%s)" % (e, time.asctime()))
        s = []
        term = False
        for i in range(0, train_size, min_batch):
            #if i % 100000 == 0:
            #    print("  %d / %d" % (i, train_size))

            # min_batch より1つ多めに入れておく
            # 学習データの最後の高々 min_batch 単語は使われない
            if train_size - i < min_batch + 1:
                term = True
            else:
                s.append(train_data[i:i + min_batch + 1])

            if len(s) == par_size or (term == True and len(s) > 0):
                model.zerograds()
                loss = model(xp.array(s, dtype=xp.int32).T)
                loss.backward()
                optimizer.update()
                s = []
            
            #if term = True:
            #    break

        out_file = out_base + ("%02d.model" % e)
        serializers.save_npz(out_file, model)
    
    print("done (%s)\n" % time.asctime())
    return

# 評価用の関数(エントロピー計算)
def calc_entropy(model):
    psum = 0.0
    wnum = 0
    ann = "# P\tlog2_P\tlog10_P\tword_ID\tword\n"
    h = model.reset()
    

    # 先頭の単語は末尾(-1)の単語から予測されるが，これは必ずEOSなので問題ない
    for i in range(test_size):
        #if i % 100000 == 0:
        #    print("  %d / %d" % (i, test_size))

        w = "%d\t%s" % (test_data[i], vocab[test_data[i]])

        # 文脈更新のために，未知語でもフォワード計算は行う
        h = model.fwd(h, xp.array([test_data[i-1]], dtype=xp.int32))

        # 未知語の結果は算入しない
        if calc_unk_pp == False and test_data[i] == unk_id:
            ann += "N/A\tN/A\tN/A\t%s\n" % (w)
            continue

        yv = F.softmax(model.W(h))
        p = yv.data[0][test_data[i]]
        lp = math.log(p, 2)
        lp10 = math.log(p, 10)
        psum -= lp
        ann += "%.5f\t%.3f\t%.3f\t%s\n" % (p, lp, lp10, w)
        wnum += 1
        
    return wnum, psum / wnum, ann


# 学習データとテストデータ
wfreq = {}
vocab = []
word2id = {}
train_data = []
test_data = []
train_in = 0
test_out = 0

#wfreq[bos] = 0
#wfreq[eos] = 0

# 学習データの読み込みと頻度計算
flist = sorted(glob.glob(train_dir + "/*.txt"))
for fname in flist:
    with open(fname, encoding='utf-8') as f:
        for line in f:
            wseq = line.split()
            if len(wseq) == 0:
                continue
            # 文頭に記号がないので追加しておく
            #train_data.append(bos)
            #wfreq[bos] += 1

            for word in wseq:
                if word not in wfreq:
                    wfreq[word] = 0
                train_data.append(word)
                wfreq[word] += 1

            # 文末に記号がないので追加しておく
            #train_data.append(eos)
            #wfreq[eos] += 1

# 単語カットオフ
for (w, c) in wfreq.items():
    if c > wcutoff:
        vocab.append(w)
        train_in += c

vocab.sort()
if unk not in vocab:
    vocab.insert(0, unk)

for i in range(len(vocab)):
    word2id[vocab[i]] = i

unk_id = 0
#bos_id = word2id[bos]
#eos_id = word2id[eos]

# 語彙ファイル保存
vocab_file = model_dir + "/" + exid + ".vocab"
with open(vocab_file, "w", encoding='utf-8') as f:
    for w in vocab:
        f.write(w + "\n")

# 単語IDへの置換
for i in range(len(train_data)):
    if train_data[i] in word2id:
        train_data[i] = word2id[train_data[i]]
    else:
        train_data[i] = unk_id

n_vocab = len(vocab)

print("---------- Data ----------")
print("Training size: %d" % len(train_data))
print("Unique words : %d" % len(wfreq))
print("Word cutoff  : %d" % wcutoff)
print("Vocab size   : %d (coverage %.2f%%)" %
      (n_vocab, train_in / len(train_data) * 100))

# テストデータの読み込み
flist = sorted(glob.glob(test_dir + "/*.txt"))
for fname in flist:
    with open(fname, encoding='utf-8') as f:
        for line in f:
            wseq = line.split()
            if len(wseq) == 0:
                continue
            # 文頭に記号がないので追加しておく
            #test_data.append(bos_id)

            for word in wseq:
                if word in word2id:
                    test_data.append(word2id[word])
                else:
                    test_data.append(unk_id)
                    test_out += 1

            # 文末に記号がないので追加しておく
            #test_data.append(eos_id)

print("\nTest size    : %d" % len(test_data))
print("Unknown words: %d (OOV rate %.2f%%)" %
      (test_out, test_out / len(test_data) * 100))

# 使用するデータ量を設定 (通常は全データ)
if args.size > 0:
    train_size = args.size
else:
    train_size = len(train_data)

test_size = len(test_data)


# 学習の実行
print("\n---------- Training ----------")
train_model(mtype, exid)

#sys.exit(0)

# 評価の実行
print("\n---------- Test (perplexity) ----------")
print("Test size: %d" % test_size)
if calc_unk_pp == True:
    print("%s will be INCLUDED in the PP calculation." % unk)
else:
    print("%s will be EXCLUDED from the PP calculation." % unk)
if args.annotate == True:
    print("Annotation files to be generated for each calculation.")
print("")

#flist1 = sorted(glob.glob(model_dir + "/" + exid + "-?.model"))
#flist2 = sorted(glob.glob(model_dir + "/" + exid + "-??.model"))
#flist = flist1 + flist2
flist = sorted(glob.glob(model_dir + "/" + exid + "-*.model"))
for fname in flist:
    model = make_model(mtype)
    serializers.load_npz(fname, model)
    cuda.get_device(0).use()
    model.to_gpu()
    wnum, ent, ann = calc_entropy(model)
    print("%s : %.2f (entropy = %.2f ) for %d words" % (fname.replace(model_dir + "/", ""), math.pow(2, ent), ent, wnum))   

    # それぞれの単語に対する確率を出力
    if args.annotate == True:
        aname = fname.replace(".model", ".ann")
        with open(aname, "w", encoding='utf-8') as f:
            f.write(ann)

print("done (%s)" % time.asctime())

## EOF
