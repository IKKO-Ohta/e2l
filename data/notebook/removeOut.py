import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../../lib/")
from load_tsvs import load_tsvs

def removeOutlier(df,corp):
    """
    すでにdfは前日比に修正済みのものとする
    """
    df_corp = df[df["COMPANY"] == corp]
    pct = df_corp[['Open', 'High', 'Low', 'Close']].pct_change()
    pct.columns = ['Open_p', 'High_p', 'Low_p', 'Close_p']
    df_corp = pd.concat([df_corp,pct],axis=1)
    
    for i in range(5):
        oStart = df_corp[df_corp["Close_p"] == df_corp.Close_p.max()].index
        oEnd = df_corp[df_corp["Close_p"] == df_corp.Close_p.min()].index
        df_corp = df_corp.drop(oStart)
        df_corp = df_corp.drop(oEnd)
    
    return df_corp

# TSVデータの読み出し
targetDir = "../stockValueFull/"
df = load_tsvs(dir_path=targetDir)
df = df.set_index("date")

for corp in df.COMPANY.unique():
    _df = removeOutlier(df,corp)
    _df["Close_p"].plot(subplots=True,figsize=(40, 10), sharex=False)
