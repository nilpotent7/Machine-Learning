import numpy as np
import random
import time
import os

x=0
error = []

def FindFiles(directory_path):
    file_paths = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def CalculateCost(result, desired):
    overall_sum = 0
    for x,y in zip(result, desired):
        overall_sum += ((x-y)*(x-y))
    return overall_sum

def CalculateError(result, desired): 
    return np.array(desired) - np.array(result)

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases  = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.position = 0

    def feedforward(self, a):
        z = sigmoid(np.dot(self.weights[self.position], a) + self.biases[self.position])
        self.position += 1
        return z

    def evaluate(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    
    def PropogateBackwards(self, input, desired_output, learning_rate):
        self.position = 0
        a1  = self.feedforward (input)
        a2  = self.feedforward (a1)
        a3  = self.evaluate    (input)
        
        a1r = a1.reshape(net.sizes[1])
        a2r = a2.reshape(net.sizes[2])
        a3r = a3.reshape(net.sizes[3])
        
        Fn1 = np.diag(np.full(net.sizes[1],(1-a1r)*a1r))
        Fn2 = np.diag(np.full(net.sizes[2],(1-a2r)*a2r))
        Fn3 = np.diag(np.full(net.sizes[3],(1-a3r)*a3r))

        s3 = np.dot(np.dot(-2, Fn3), CalculateError(a3, desired_output))
        s2 = np.dot(np.dot(Fn2, np.transpose(self.weights[2])), s3)
        s1 = np.dot(np.dot(Fn1, np.transpose(self.weights[1])), s2)

        self.weights[2] = self.weights[2] - np.dot(np.dot(learning_rate, s3), np.transpose(a2))
        self.weights[1] = self.weights[1] - np.dot(np.dot(learning_rate, s2), np.transpose(a1))
        self.weights[0] = self.weights[0] - np.dot(np.dot(learning_rate, s1), np.transpose(input))
        self.biases[2]  = self.biases[2]  - np.dot(learning_rate, s3)
        self.biases[1]  = self.biases[1]  - np.dot(learning_rate, s2)
        self.biases[0]  = self.biases[0]  - np.dot(learning_rate, s1)

        print(self.weights)
        print(self.biases)

    def SaveData(self, path):
        for k,x in enumerate(self.weights):
            np.save(path+f"\\weights{k}", x)
        for k,x in enumerate(self.biases):
            np.save(path+f"\\biases{k}", x)

    def LoadDataVar(self, b, w):
        self.weights = w
        self.biases = b

    def LoadData(self, path):
        allFiles = FindFiles(path)
        weightsFiles = [x for x in allFiles if "weights" in x]
        biasesFiles = [x for x in allFiles if "biases" in x]
        self.weights = [np.load(w) for w in weightsFiles]
        self.biases = [np.load(b) for b in biasesFiles]

def SetupDesiredOutputs(I, DN, C):
    DO = [0] * len(I)
    for x,i in enumerate(I):
        if():   DO[x] = DN[0]
        if():   DO[x] = DN[1]
        if():   DO[x] = DN[2]
        if():   DO[x] = DN[2]
        if():   DO[x] = DN[2]
        if():   DO[x] = DN[2]
    return DO

net = Network([3, 2, 2, 1])

net.LoadData("GPU Version\\TestingData")

Inputs = [
    np.array([1, 0, 1]).reshape(3,1).astype(np.float64)
]
desired_outputs = np.array([1]).reshape(1,1).astype(np.float64)


net.PropogateBackwards(Inputs, desired_outputs, 0.1)