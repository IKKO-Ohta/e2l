import os
import pickle
import glob

class myLoader:
    def __init__(self):
        self.targetDir = ""
        self.dir_cnt = 0
        self.file_cnt = 1
        self.sentence_cnt = 0
        self.step = 0
        self.train = 0
        self.test = 0

    def set(self, d="../auto/preprocessed/", train=23, test=24):
        if d != "../auto/preprocessed/":
            self.targetDir = d

        self.dirs = glob.glob(self.targetDir + "*/*.pkl")
        self.train = train
        self.test = test

    def myNameFormat(self):
        path = '{0:02d}'.format(self.targetDir) + "/" \
                + "wsj" + "_" \
                + '{0:04d}'.format(self.file_cnt) \
                +  str(self.sentence_cnt) \
                + '{0:07d}'.format(self.file_cnt) + ".pkl"
        return path

    def findNextSentence(self):
        self.step = 0
        self.sentence_cnt += 1
        candidate = self.myNameFormat(self)
        if os.path.exists(candidate):
            return candidate
        else:
            return ""

    def findNextFile(self):
        self.step = 0
        self.sentence_cnt = 0
        self.file_cnt += 1
        candidate = self.myNameFormat(self)
        if os.path.exists(candidate):
            return candidate
        else:
            return ""

    def findNextDir(self):
        self.step = 0
        self.sentence_cnt = 0
        self.file_cnt = 0
        self.dir_cnt += 1
        if self.dir_cnt == self.test:
            print("note:",self.dir, "for test")
            print("Next Dir")
            self.dir_cnt += 1

        candidate = self.myNameFormat(self)
        if os.path.exists(candidate):
            return candidate
        else:
            return ""

    def gen(self):
        """
        return:
        [([...],[label]),
         ([...],[label]),
         (),()...]
        もしイテレータが尽きていたら例外を送出する
        """
        result = []

        while(1):
            target_path = myNameFormat(self)
            if not os.path.exists(target_path):
                break

            with open(target_path) as f:
                pkl = pickle.load(f)
                result.append(pkl)

            self.step += 1

        # 1 sentenceを読み終わるタイミング
        if self.findNextSentence(target_path):
            print("Next:", self.findNextSentence(target_path))
        else:
            if self.findNextFile(target_path):
                print("Next:", self.findNextFile(target_path))
            else:
                if self.findNextDir(target_path):
                    print("Next:", self.findNextDir(target_path)):
                else:
                    # 全てのファイルを読み終わる
                    raise IndexError

        # 次があることを保証してからreturn
        assert(os.path.exists(myNameFormat(self)))
        return result
