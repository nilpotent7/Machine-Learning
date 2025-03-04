import numpy as np
from PIL import Image
import os
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def FindFiles(directory_path):
    file_paths = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

def OutputResult(output, characterMap):
    for k,o in enumerate(output):
        print(f"{characterMap[k]} : {o}")

def ReadImage(fileName):
    im = Image.open(fileName, 'r')
    return [x[0] / 255 for x in list(im.getdata())]

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases  = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.position = 0

    def feedforward(self, a, rounded):
        if(a.shape != (self.sizes[0],1)):
            raise Exception("Input array shape does not correspond to neurons in input layer.")
        
        o = sigmoid(np.dot(self.weights[self.position], a) + self.biases[self.position])
        self.position += 1
        if rounded: o = np.round(o, decimals=2)
        return o

    def evaluate(self, a, rounded):
        if(a.shape != (self.sizes[0],1)):
            raise Exception("Input array shape does not correspond to neurons in input layer.")
        
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        if rounded: o = np.round(a, decimals=5)
        else: o = a
        return o

    def LoadDataVar(self, b, w):
        self.weights = w
        self.biases = b
    
    def LoadData(self, path):
        allFiles = FindFiles(path)
        weightsFiles = [x for x in allFiles if "weights" in x]
        biasesFiles = [x for x in allFiles if "biases" in x]
        self.weights = [np.load(w) for w in weightsFiles]
        self.biases = [np.load(b) for b in biasesFiles]

c = ["+", "-", "\\", "x"]

Inputs = [
    np.array([
        0,1,0,
        1,1,1,
        0,1,0
    ]).reshape((9,1)),

    np.array([
        0,0,0,
        1,1,1,
        0,0,0
    ]).reshape((9,1)),

    np.array([
        1,0,0,
        0,1,0,
        0,0,1
    ]).reshape((9,1)),

    np.array([
        1,0,1,
        0,1,0,
        1,0,1
    ]).reshape((9,1))
]

net = Network([9, 16, 16, 4])
net.LoadData("Model\\Data")

# print("Model Error: " + str(CalculateCost(net, Inputs, DesiredOutputs)))
OutputResult(net.evaluate(Input, True), c)