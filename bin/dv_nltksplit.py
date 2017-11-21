import nltk.data

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
fp = open("../auto/djnml_daily_headline/20120702.txt")
data = fp.read()
EOSs = tokenizer.tokenize(data)
print(EOSs)
