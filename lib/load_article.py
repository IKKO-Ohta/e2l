import sys
import os
def load_article(file_path,convert=True):
    def datetimeToFilepath(d):
        path = "../auto/djnml_daily_headline_splited/"+d+".txt"
        if os.path.exists(path):
            return path
        else:
            return ""

    if convert == True:
        file_path = datetimeToFilepath(file_path)
        if file_path == "": return 0


    result = []
    with open(file_path) as f:
        for line in f:
            #line = line.rstrip()
            result.append(line)
    return result

if __name__ == '__main__':
    print(load_article(sys.argv[1]))
