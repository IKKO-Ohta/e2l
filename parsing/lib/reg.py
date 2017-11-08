import regex

def id2wv(Id, id2word, w2vModel):
    """
    param: 単語ID, ID => word辞書 ,word2vec辞書
    return: word2vec prefix vector
    """
    word = id2word[Id]
    myreg = re.compile('[a-zA-Z0-9]+')
    g = self.regex.match(word)
    word = g.group()
    if word in w2vModel:
        return w2vModel[word]
    elif word.capitalize() in w2vModel:
        return w2vModel[word.capitalize()]
    else:
        print("w2v not found. =>", word)
        return np.asarray([0 for i in range(300)],dtype=np.float32)
