import pandas as pd
import glob
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def MyLoad(file_pathes):
    
    def mydate(filepath):
        s = filepath.split("/")[-1]
        return s.replace(".tsv","")

    for file_path in file_pathes:
        if file_path == file_pathes[0]:
            df = pd.read_csv(file_path,delimiter="\t",header=None,
             columns=["COMPANY","Open","High","Low","Close"])
            df["date"] = mydate(file_path)
        else:
            _df = pd.read_csv(file_path,delimiter="\t",header=None,
             columns=["COMPANY","Open","High","Low","Close"])
            _df["date"] = mydate(file_path)

            df = pd.concat(df,_df,axis=1)

    df = df[["date","COMPANY","Open","High","Low","Close"]]
    df.reindex()
    return df

if __name__ == '__main__':
    file_pathes = glob.glob("../auto/dj39/*")
    print(myload(file_pathes))
