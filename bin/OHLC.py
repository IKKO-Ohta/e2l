import pandas as pd
import glob
from collections import OrderedDict
####
# AA 2013 09 25 付近で明らかな異常値を発見。調査が必要
###

def preprocess(line):
    '''                                                                                               
    AA  20050103    0933    31.67   14100   23                                                        
    ->['AA', '20050103', '0933', '31.67', '14100', '23']                                              
    '''
    line = line.strip().split('\t')
    return line

def myappend(d,line):
    try:
        d[line[1]].append(line[3])
    except:
        d[line[1]] = []
        d[line[1]].append(line[3])
    return

if __name__ == '__main__':
    
    file_pathes = glob.glob("../resource/numerical/dow30/*.txt")
    days = {}
    c_names = [file_path.split("/")[-1].replace(".2005-2014.txt","") for file_path in file_pathes]                                                      

    for file in file_pathes:
        f = open(file,'r')
        price_dict = OrderedDict()  # 初期化   
        features = []

        for line in f: # 会社当たりの辞書を作成                                                      
            line = preprocess(line)
            if ('20120701' <= line[1] <= '20150630'):                
                myappend(price_dict,line)

        for k,v in price_dict.items():
            O,H,L,C = v[0],max(v),min(v),v[-1]
            #print(k,[O,H,L,C])
            try:
                days[k].append([O,H,L,C])
            except:
                days[k] = []
                days[k].append([O,H,L,C])

        print('working...\n')
    
        for datetime,values in days.items():
            with open("../auto/dj39/"+datetime+".tsv","w") as f:
                for name,company in zip(c_names,values):
                    line = company[:]
                    line.insert(0,name)
                    print(line)
                    S = "\t".join(line)+"\n"
                    f.write(S)