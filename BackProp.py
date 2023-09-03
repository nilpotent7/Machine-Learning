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
    e = np.array(desired) - np.array(result)    
    return e

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
        a2  = self.feedforward (input)
        a1  = self.evaluate    (input)
        
        a1r = a1.reshape(net.sizes[2])
        a2r = a2.reshape(net.sizes[1])
        
        Fn  = np.diag(np.full(net.sizes[2],(1-a1r)*a1r))
        Fn2 = np.diag(np.full(net.sizes[1],(1-a2r)*a2r))

        s2 = np.dot(np.dot(-2, Fn), CalculateError(a1, desired_output))
        s1 = np.dot(np.dot(Fn2, np.transpose(self.weights[1])), s2)

        self.weights[1] = self.weights[1] - np.dot(np.dot(learning_rate, s2), np.transpose(a2))
        self.weights[0] = self.weights[0] - np.dot(np.dot(learning_rate, s1), np.transpose(input))
        self.biases[1]  = self.biases[1]  - np.dot(learning_rate, s2)
        self.biases[0]  = self.biases[0]  - np.dot(learning_rate, s1)

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

net = Network([4, 10, 4])

#net.LoadData("Data")

inputs = [
    np.array([1, 0, 1, 0]).reshape(4,1),
    np.array([1, 0, 0, 1]).reshape(4,1),
    np.array([0, 1, 1, 0]).reshape(4,1),
    np.array([0, 0, 1, 1]).reshape(4,1),
    np.array([0, 1, 0, 1]).reshape(4,1),
    np.array([1, 1, 0, 0]).reshape(4,1)
]

desired_outputs = [
    np.array([1, 0, 0, 0]).reshape(4,1),
    np.array([0, 1, 0, 0]).reshape(4,1),
    np.array([0, 0, 1, 0]).reshape(4,1),
    np.array([0, 0, 0, 1]).reshape(4,1),
    np.array([1, 0, 0, 0]).reshape(4,1),
    np.array([0, 0, 0, 1]).reshape(4,1)
]

s = time.time()
while x < 500:
    k = random.randint(0, 5)
    y = 0
    while y<500:
        net.PropogateBackwards(inputs[k], desired_outputs[k], 0.1)
        y+=1
    x+=1
print("Finished in " + str(time.time()-s))

net.SaveData("Models\\Data")