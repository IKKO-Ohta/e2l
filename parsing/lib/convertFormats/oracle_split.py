import sys

def mywrite(elems, cnt):
    path = "testOracleSplited/ud-test-"+ str(cnt) + ".oracle"
    with open(path, "w") as p:
        for elem in elems:
            p.write(elem)

def mysplit(path):
    cnt = 0
    result = []
    with open(path,"r") as f:
        for line in f:
            #print(line)
            result.append(line)
            if line == "[ROOT-ROOT][]\n":
                mywrite(result, cnt)
                cnt += 1
                result = []
    return

if __name__ == '__main__':
    if len(sys.argv) == 2:
        path = sys.argv[1]
    else:
        path = "../../auto/UD_conll/devOracle.txt"
    mysplit(path)
    print("done")
