
"""
lstm-parserが健全に動作するかどうかを判定する。
データセットPenn_Oracle_split_test/[00,24]/*.oracleを適切に学習して返せるかどうかを試す。
中間表現はauto/test/[00,24]/*.pklとして保存する。
"""
import subuprocess
import sys
import glob
from oracle2vec import Configuration
from oracle2vec import Transition
from oracle2vec import myVectorizer
from lstmParser import Parser
from lstmParser import evaluate, backupModel, composeTensor

def pklComile():
    """
    auto/Penn_Oracle_split_test/wsj_0001_0.oracle
    => auto/preprocessed/wsj_0001_0_[step数].pkl
    の写像
    ここでoracleread.pyを読む
    """
    pathes = glob.glob("../auto/Penn_Oracle_split/*/*.oracle")
    vectorizer = myVectorizer()
    error = 0

    for path in pathes:
        words, actions = oracle_load(path)
        conf = Configuration()
        conf.stack = copy.deepcopy(words[0][0])
        conf.buffer = copy.deepcopy(words[0][1])
        cnt = 0
        try:
            for action in actions:
                his = vectorizer.cal_history(conf.history)
                buf = vectorizer.buf_embed(conf.buffer)
                stk = vectorizer.edge_embed(conf.arcs)
                if action == "SHIFT":
                    Transition.shift(conf)
                elif "RIGHT" in action:
                    Transition.right_arc(action, conf)
                elif "LEFT" in action:
                    Transition.left_arc(action, conf)
                else:
                    sys.stderr("!! 予期しない命令が発生しました..")
                    break
                target_dir_num = path.split("/")[-2]
                origin_name = path.split("/")[-1].replace(".oracle", "")
                target_path = "../auto/preprocessed/" + target_dir_num \
                              + "/" + origin_name + "_" + '{0:07d}'.format(cnt) + ".pkl"

                with open(target_path, "wb") as target:
                    print("gen:", target_path)
                    label = vectorizer.act_map[action]
                    pickle.dump(([his, buf, stk], label), target)

                conf.history.append(action)
                cnt += 1
        except IndexError:
            print("--ERROR-- ", path)
            print("IndexError")
            print("continue..")
            error += 1
    return

def trainParser():
    """
    parser自体を駆動する。
    同時にloaderも回転させる。
    """
    loader = myLoader()
    loader.set()
    model = Parser()
    optimizer = optimizers.SGD()
    optimizer.setup(model)
    model.reset_state()
    model.cleargrads()
    print("loading..")
    hisTensor, bufTensor, stkTensor, testMat = composeTensor(loader,model)
    print("loaded!")

    timecnt = 0
    for epoch in range(1):
        epochTimeStart = time.time()
        for hisMat, bufMat, stkMat, testVec in zip(hisTensor, bufTensor, stkTensor,testMat):
            loss = model(hisMat, bufMat, stkMat, testVec)
            loss.backward()
            optimizer.update()
            model.reset_state()

            timecnt += 1
            if timecnt % 100 == 0:
                print("■", end="")

        print("epoch",epoch,"finished..")
        print("epoch time:{0}".format(time.time()-epochTimeStart))
        print("backup ..")
        backupModel(model, epoch)
        print("evaluate ..")
        evaluate(model,loader)
        print("Next epoch..")

    print("finish!")
    modelName = "parserModel" + str(datetime.datetime.now())
    serializers.save_hdf5("../model/"+"complete_"+modelName, model)
    return model,loader

if __name__ == '__main__':
    print("pklCompile..")
    pklCompile()
    print("trainParser..")
    letModel,letLoader = trainParser()
    print("testEvaluate..")
    evaluate(letModel,letLoader)
    print("..OK!")
