import sys
import pandas as pd
def load_tsv(f,convert=True):
    """
    param:
        f: YYYYMMDD
        convert: 日付=>ファイル変換をするか。もしFalseにしたらfには直接パスを渡すこと
    return:
        ファイルパスのtsvを読み出し、カラムをつけてconcatしたもの
        カラムは、["date","COMPANY","Open","High","Low","Close"]
    """
    def datetimeToFilepath(d):
        return "../auto/dj39/"+d+".tsv"

    def mydate(file_path):
        s = file_path.split("/")[-1]
        return s.replace(".tsv","")

    if convert == True:
        f = datetimeToFilepath(f)

    df = pd.read_csv(f,delimiter="\t",header=None,
        names=["COMPANY","Open","High","Low","Close"])
    df["date"] = mydate(f)

    df = df[["date","COMPANY","Open","High","Low","Close"]]
    df.reindex()
    return df

if __name__ == '__main__':
    print(load_tsv(sys.argv[1]))
