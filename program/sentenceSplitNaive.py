import glob
import sys
sys.path.append("../lib")
from load_article import load_article

def splitEOS(sentence):
    return sentence.split(".")

def mysplit(filePath):
    lines = load_article(filePath,convert=False)
    fileName = filePath.split("/")[-1]
    targetPath = "../auto/djnml_daily_headline_splited/" + fileName
    with open(targetPath,"w") as f:
        for line in lines:
            sentences = splitEOS(line) # 文区切りの処理
            for sentence in sentences:
                if not "\n" in sentence:
                    f.write(sentence+"\n")
    return

if __name__ == '__main__':
    filesPath = glob.glob("../auto/djnml_daily_headline/*.txt")
    for filePath in filesPath:
        print("split: ",filePath)
        mysplit(filePath)
