import glob
import os

"""
2つのディレクトリの日付が完全に等しいことをvalidateする。
もし差異があったら、積集合だけを残すように削除する。
"""

articles = glob.glob("../train/*.txt")
stockValues = glob.glob("../stockValueCloseTrain/*.tsv")

articles = [article.replace("../train/","").replace(".txt","")[:-5] for article in articles]
stockValues = [sv.replace("../stockValueCloseTrain/","").replace(".tsv","") for sv in stockValues]

interSection = set(articles).intersection(set(stockValues))
print("intersection:",len(interSection),"days,")

left = set(articles) - set(stockValueTrain)
right = set(stockValueTrain) - set(articles)

for l in left:
    delFiles = ("../train/"+l+"*")
    [os.remove(delFile) for delFile in delFiles]
for r in right:
    os.remove("../stockValueCloseTrain"+ r + ".tsv")
