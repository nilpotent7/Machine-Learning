import numpy as np
import os

def FindFiles(directory_path):
    file_paths = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

Path = input()
Data = [np.load(x) for x in FindFiles(Path)]
print(Data[-1][1:])