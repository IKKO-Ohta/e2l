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
    targetPath = "../auto/article_part_splited/" + fileName
    with open(targetPath,"w") as f:
        for line in result:
            line = line + "\n" if line != "." else "\n"
            f.write(line)
    return

if __name__ == '__main__':
    filesPath = glob.glob("../auto/article_part_body/*.txt")
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    for filePath in filesPath:
        print("split: ",filePath)
        mysplit(filePath,tokenizer)
