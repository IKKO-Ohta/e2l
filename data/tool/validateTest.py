import glob
import os

"""
2つのディレクトリの日付が完全に等しいことをvalidateする。
もし差異があったら、積集合だけを残すように削除する。
"""

articles = glob.glob("../test/*.txt")
stockValues = glob.glob("../stockValueCloseTest/*.tsv")

articles = [article.replace("../test/","").replace(".txt","")[:-5] for article in articles]
stockValues = [sv.replace("../stockValueCloseTest/","").replace(".tsv","") for sv in stockValues]

interSection = set(articles).intersection(set(stockValues))
print("intersection:",len(interSection),"days,")

left = set(articles) - set(stockValuetest)
right = set(stockValuetest) - set(articles)

for l in left:
    delFiles = ("../test/"+l+"*")
    [os.remove(delFile) for delFile in delFiles]
for r in right:
    os.remove("../stockValueCloseTest"+ r + ".tsv")
