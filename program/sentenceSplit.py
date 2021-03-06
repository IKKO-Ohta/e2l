import glob
import sys
import nltk

def splitEOS(sentence):
    return sentence.split(".")

def mysplit(filePath,tk):
    with open(filePath) as f:
        txt = f.read()
    result = tk.tokenize(txt)
    fileName = filePath.split("/")[-1]
    targetPath = "../data/auto/djnml_daily_headline_splited/" + fileName
    with open(targetPath,"w") as f:
        for line in result:
            line = line + "\n" if len(line) > 3  else "\n"
            f.write(line)
    return

if __name__ == '__main__':
    filesPath = glob.glob("../data/auto/origIshino/article_part_body/*.txt")
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    for filePath in filesPath:
        print("split: ",filePath)
        mysplit(filePath,tokenizer)
