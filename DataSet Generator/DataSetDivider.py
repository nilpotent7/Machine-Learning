import os
import random
import shutil

def FindFiles(directory_path):
    file_paths = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

allFiles  = FindFiles("Symbols\\Complete")

dollar   = [x for x in allFiles if "$" in x]
multi    = [x for x in allFiles if "x" in x]
sigma    = [x for x in allFiles if "∑" in x]
percent  = [x for x in allFiles if "%" in x]
atsign   = [x for x in allFiles if "@" in x]

random.shuffle(dollar)
random.shuffle(sigma)
random.shuffle(multi)
random.shuffle(percent)
random.shuffle(atsign)

testing_dollar   = dollar   [-2:]
testing_sigma    = sigma    [-2:]
testing_multi    = multi    [-2:]
testing_percent  = percent  [-2:]
testing_atsign   = atsign   [-2:]

dollar   = dollar  [:-2]
sigma    = sigma   [:-2]
multi    = multi   [:-2]
percent  = percent [:-2]
atsign   = atsign  [:-2]

allFiles=[]
allFiles.extend(dollar)
allFiles.extend(sigma)
allFiles.extend(multi)
allFiles.extend(percent)
allFiles.extend(atsign)

testingFiles=[]
testingFiles.extend(testing_dollar)
testingFiles.extend(testing_sigma)
testingFiles.extend(testing_multi)
testingFiles.extend(testing_percent)
testingFiles.extend(testing_atsign)

newDest1 = "Symbols\\Training"
newDest2 = "Symbols\\Testing"


for f in allFiles:
    shutil.move(f, f.replace("Symbols\\Complete", newDest1))

for f in testingFiles:
    shutil.move(f, f.replace("Symbols\\Complete", newDest2))