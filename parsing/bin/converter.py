import subprocess
import glob

def myname(path):
    path = path.split("/")
    d,f = path[-2],path[-1]
    return "../auto/Penn_conbined_wsj/" + d + "/" + f

files = glob.glob("../corpus/Penn_conbined_wsj/*/*")
converter = "java -jar /mnt/mqs02/home/ikko/pennconverter/pennconverter-1398335914.jar"

for file in files:
    arg = [converter, "<", file, ">", myname(file)]; print(" ".join(arg))
    subprocess.run(" ".join(arg),shell=True)
