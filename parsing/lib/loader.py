import os
import pickle
import glob

class myLoader:
    def __init__(self):
        self.targetRoot = ""
        self.dir_cnt = 0
        self.file_cnt = 1
        self.sentence_cnt = 0
        self.step = 0
        self.train = 0
        self.test = 0

    def set(self, d="../auto/preprocessed/", train=23, test=24):
        self.targetRoot = d
        self.train = train
        self.test = test

    def loderState(self):
        print("self.dir_cnt", self.step)
        print("self.file_cnt", self.step)
        print("self.sentence_cnt", self.sentence_cnt)
        print("self.step",self.step)

    def myNameFormat(self):
        path =  self.targetRoot \
                +'{0:02d}'.format(self.dir_cnt) + "/" \
                + "wsj" + "_" \
                + '{0:04d}'.format(self.file_cnt) + "_" \
                +  str(self.sentence_cnt)+ "_" \
                + '{0:07d}'.format(self.step) + ".pkl"
        return path

    def findNextSentence(self):
        self.step = 0
        self.sentence_cnt += 1
        candidate = self.myNameFormat()
        if os.path.exists(candidate):
            return candidate
        else:
            return ""

    def findNextFile(self):
        self.step = 0
        self.sentence_cnt = 0
        self.file_cnt += 1
        candidate = self.myNameFormat()
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
            print("Note:",self.dir_cnt, "is reserved for test Directory")
            print("Next dir..")
            self.dir_cnt += 1

        candidate = self.myNameFormat()
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
            target_path = self.myNameFormat()
            if not os.path.exists(target_path):
                break

            with open(target_path,"rb") as f:
                pkl = pickle.load(f)
                print("Load:", target_path)
                result.append(pkl)

            self.step += 1

        # 1 sentenceを読み終わるタイミング
        if self.findNextSentence():
            print("Next:", self.myNameFormat())
        else:
            if self.findNextFile():
                print("Next:", self.myNameFormat())
            else:
                if self.findNextDir():
                    print("Next:", myNameFormat())
                else:
                    # 全てのファイルを読み終わる
                    raise IndexError

        # 次があることを保証してからreturn
        assert(os.path.exists(self.myNameFormat()))
        return result

if __name__ == '__main__':
    loader = myLoader()
    loader.set()
    print(loader.gen())
    print(loader.gen())
