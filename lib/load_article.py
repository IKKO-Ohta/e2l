import sys

def load_article(file_path,convert=True):
    def datetimeToFilepath(d):
        return "../auto/djnml_daily_headline/"+d+".txt"

    if convert == True:
        file_path = datetimeToFilepath(file_path)

    result = []
    with open(file_path) as f:
        for line in f:
            #line = line.rstrip()
            result.append(line)
    return result

if __name__ == '__main__':
    print(load_article(sys.argv[1]))
