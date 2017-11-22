import pickle
import glob

def buildVocab(filePathes):
    vocab,Id = {},0

    for filePath in filePathes:
        with open(filePath) as f:
            for line in f:
                line = line.replace(".\n","").replace("\n","").split(" ")
                for elem in line:
                    if not elem in vocab:
                        vocab[elem] = Id
                        Id += 1
    vocab["<EOS>"] = Id + 1
    return vocab

if __name__ == '__main__':
    articles = glob.glob("../auto/djnml_daily_headline_splited/*.txt")
    word2id = buildVocab(articles)
    id2word = {v:k for k,v in word2id.items()}
    with open("pkls/word2id.pkl","wb") as f:
        pickle.dump(word2id,f)
    with open("pkls/id2word.pkl","wb") as f:
        pickle.dump(id2word,f)
    print("Generated! pkl_len: ",len(word2id))
