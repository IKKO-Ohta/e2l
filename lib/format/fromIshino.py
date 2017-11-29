import glob

def solve(path):
    result = []
    with open(path) as f:
        flag = False
        for line in f:
            if "<text>" in line:
                flag = True
            elif "</text>" in line:
                flag = False

            if flag:
                line = line[1:]
                result.append(line)
    fileName = filePath.split(".")[-1].replace(".txt","")
    with open("../../auto/article_part_body/"+fileName+".txt","w") as f:
        for res in result:
            f.write(res)
    return

if __name__ == '__main__':
    filePathes = glob.glob("../../auto/article_part/*.txt")
    for filePath in filePathes:
        solve(filePath)
