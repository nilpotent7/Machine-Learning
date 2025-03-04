import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def FindFiles(directory_path):
    file_paths = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

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

        A = [input]
        k = 1
        while k < len(self.sizes):
            A.append(self.feedforward (A[k-1]))
            k+=1
        A.pop(0)
        
        AR = [x.reshape(self.sizes[k+1]) for k,x in enumerate(A)]
        Fn = [np.diag((1 - x) * x) for _,x in enumerate(AR)]
        
        S = [2 * np.dot(Fn[-1], CalculateError(A[-1], desired_output))]
        k = 2
        while k < len(self.sizes):
            Fni = Fn[-k]
            Wi  = self.weights[len(self.sizes)-k]
            Si  = S[k-2]
            S.append(np.dot(np.dot(Fni, np.transpose(Wi)), Si))
            k+=1

        k = 1
        while k < len(self.sizes):
            Wi = self.weights[len(self.sizes)-k-1]
            Bi = self.biases [len(self.sizes)-k-1]
            Si = S[k-1]
            if -(k+1) == -(len(self.sizes)): Ai = input
            else: Ai = A[-(k+1)]
            self.weights[len(self.sizes)-k-1] = Wi - np.dot(np.dot(learning_rate, Si), np.transpose(Ai))
            self.biases [len(self.sizes)-k-1] = Bi - np.dot(learning_rate, Si)
            k+=1

    def SaveData(self, path):
        os.makedirs(path, exist_ok=True)
        for k, x in enumerate(self.weights):
            np.save(os.path.join(path, f"weights{k}.npy"), x)
        for k, x in enumerate(self.biases):
            np.save(os.path.join(path, f"biases{k}.npy"), x)
    
    def SaveImage(self, path):
        os.makedirs(path, exist_ok=True)
        
        for k, x in enumerate(self.weights):
            plt.imshow(x, cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.title(f'Weights {k}')
            plt.savefig(os.path.join(path, f"weights{k}.png"))
            plt.close()
        
        for k, x in enumerate(self.biases):
            plt.figure(figsize=(len(x), 1))
            plt.imshow(x.reshape(1, -1), cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.title(f'Biases {k}')
            plt.savefig(os.path.join(path, f"biases{k}.png"))
            plt.close()

    def LoadDataVar(self, b, w):
        self.weights = w
        self.biases = b

    def LoadData(self, path):
        allFiles = FindFiles(path)
        weightsFiles = [x for x in allFiles if "weights" in x]
        biasesFiles = [x for x in allFiles if "biases" in x]
        self.weights = [np.load(w) for w in weightsFiles]
        self.biases = [np.load(b) for b in biasesFiles]

def CalculateError(result, desired): 
    e = np.array(result) - np.array(desired)
    return e

def CalculateCost(Net, TestingSet, DesiredTSet):
    SumOfSum = 0
    Total = 0
    
    for k,x in enumerate(TestingSet):
        desired = DesiredTSet[k]
        result = Net.evaluate(x)
        
        overall_sum = 0
        for x,y in zip(result, desired):
            overall_sum += ((x-y)*(x-y))
        
        SumOfSum += overall_sum
        Total +=1

    return SumOfSum / Total

net = Network([9, 16, 16, 4])

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

DesiredOuputs = [
    np.array([
        1,0,0,0
    ]).reshape((4,1)),

    np.array([
        0,1,0,0
    ]).reshape((4,1)),

    np.array([
        0,0,1,0
    ]).reshape((4,1)),

    np.array([
        0,0,0,1
    ]).reshape((4,1)),
]

print("Now Learning!\n\n")

epochs = 250

for e in tqdm(range(epochs), desc="Epoch Progress"):
    for x in range(len(Inputs)):
        for q in range(100):
            net.PropogateBackwards(Inputs[x], DesiredOuputs[x], 0.1)

print("Model Error: " + str(CalculateCost(net, Inputs, DesiredOuputs)))
print("Finished Learning!")
net.SaveData("Model\\Data")
net.SaveImage("Model\\VisualData")