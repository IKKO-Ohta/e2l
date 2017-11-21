import sys
import os
import pandas as pd
def load_tsv(f,convert=True,feature=False):
    """
    param:
        f: YYYYMMDD
        convert: 日付=>ファイル変換をするか。もしFalseにしたらfには直接パスを渡すこと
    return:
        ファイルパスのtsvを読み出し、カラムをつけてconcatしたもの
        カラムは、["date","COMPANY","Open","High","Low","Close"]
    """
    def datetimeToFilepath(d):
        path = "../auto/dj39/"+d+".tsv"
        if os.path.exists(path):
            return path
        else:
            return ""

    def mydate(file_path):
        s = file_path.split("/")[-1]
        return s.replace(".tsv","")

    if convert == True:
        f = datetimeToFilepath(f)
        if f == "": return 0

    df = pd.read_csv(f,delimiter="\t",header=None,
        names=["COMPANY","Open","High","Low","Close"])
    df["date"] = mydate(f)

    df = df[["date","COMPANY","Open","High","Low","Close"]]
    df.reindex()
    if feature == False:
        return df
    else:
        return df[["Open","High","Low","Close"]]

if __name__ == '__main__':
    print(load_tsv(sys.argv[1]))
