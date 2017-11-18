import pandas as pd
import glob
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def load_tsvs(dir_path="../auto/dj39/"):
    """
    param: ディレクトリ パス, デフォルトでは"../auto/dj39/"
    return: ディレクトリのtsvを全て読み出し、カラムをつけてconcatしたもの
    カラムは、["date","COMPANY","Open","High","Low","Close"]
    """
    import glob
    import pandas as pd
    file_pathes = glob.glob(dir_path+"*.tsv")
    def mydate(file_path):
        s = file_path.split("/")[-1]
        return s.replace(".tsv","")

    for file_path in file_pathes:
        if file_path == file_pathes[0]:
            df = pd.read_csv(file_path,delimiter="\t",header=None,
                names=["COMPANY","Open","High","Low","Close"])
            df["date"] = mydate(file_path)
        else:
            _df = pd.read_csv(file_path,delimiter="\t",header=None,
                names=["COMPANY","Open","High","Low","Close"])
            _df["date"] = mydate(file_path)

            df = pd.concat([df,_df],axis=0)

    df = df[["date","COMPANY","Open","High","Low","Close"]]
    df.reindex()
    return df

if __name__ == '__main__':
    print(load_tsvs())
