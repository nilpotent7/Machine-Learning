from PIL import Image
import numpy as np
import random
import time
import os


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
        
        AR = [x.reshape(net.sizes[k+1]) for k,x in enumerate(A)]
        Fn = [np.diag(np.full(net.sizes[k+1],(1-x)*x)) for k,x in enumerate(AR)]
        
        S = [np.dot(np.dot(-2, Fn[-1]), CalculateError(A[-1], desired_output))]
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

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def CalculateError(result, desired): 
    e = np.array(desired) - np.array(result)    
    return e

def CalculateCost(Net, TestingSet, DesiredTSet):
    SumOfSum = 0
    Total = 0
    
    for k,x in enumerate(TestingSet):
        desired = DesiredTSet[k]
        result = Net.evaluate(np.array(ReadImage(x)).reshape(2500,1))
        
        overall_sum = 0
        for x,y in zip(result, desired):
            overall_sum += ((x-y)*(x-y))
        
        SumOfSum += overall_sum
        Total +=1

    return SumOfSum / Total

def FindFiles(directory_path):
    file_paths = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

def ReadImage(fileName):
    im = Image.open(fileName, 'r')
    return [x[0] / 255 for x in list(im.getdata())]



def SetupDesiredNeurons():
    l = []
    l.append(np.array([1, 0, 0]).reshape(3,1))
    l.append(np.array([0, 1, 0]).reshape(3,1))
    l.append(np.array([0, 0, 1]).reshape(3,1))
    return l

def SetupDesiredOutputs(I, DN, C):
    DO = [0] * len(I)
    for x,i in enumerate(I):
        if(C[0]  in i):   DO[x] = DN[0]
        if(C[1]  in i):   DO[x] = DN[1]
        if(C[2]  in i):   DO[x] = DN[2]
    return DO


Symbols = ["x", "+", "%"]
BatchSize = 1000

Inputs  = FindFiles("DataSet\\Symbols Large\\Training")
Testing = FindFiles("DataSet\\Symbols Large\\Testing")
random.shuffle(Inputs)
random.shuffle(Testing)

DesiredNeurons = SetupDesiredNeurons()
DesiredOuputs  = SetupDesiredOutputs(Inputs,  DesiredNeurons, Symbols)
DesiredOuputsT = SetupDesiredOutputs(Testing, DesiredNeurons, Symbols)

Inputs = [Inputs[i:i + BatchSize] for i in range(0, len(Inputs), BatchSize)]

#net = Network([2500, 128, 128, 128, 3])
net = Network([10, 8, 8, 5])
net.SaveData("TestingData")
exit()

os.system("cls")
print(f"Initial Cost of the network: {CalculateCost(net, Testing, DesiredOuputsT)}")

input("Hit enter to start learning...")
print("Now Learning!\n\n")

Fs = time.time()

e = 0
while e < 10:
    s2 = time.time()

    for x,batch in enumerate(Inputs):
        s = time.time()
        
        for y,inputPath in enumerate(batch):
            inputImage = np.array(ReadImage(inputPath)).reshape(2500,1)
            q=0
            while q<10:
                net.PropogateBackwards(inputImage, DesiredOuputs[(x*BatchSize)+y], 0.1)
                q+=1
        print(f"Loop {e+1} | Batch {x+1} | Time Took: {str(round(time.time()-s))}s | Cost: {CalculateCost(net, Testing, DesiredOuputsT)}")
    
    print(f"Loop Finished! Took {round(time.time() - s2)}s")
    net.SaveData("Model\\Data4")
    e+=1    

print("Finished in " + str(time.time()-Fs))

input("Hit enter to save weights and biases...")
net.SaveData("Model\\Data4")