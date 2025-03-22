import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(a):
    return a * (1 - a)

def FindFiles(directory_path):
    file_paths = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

def LoadDataset(path="dataset.npz"):
    data = np.load(path)
    train_images = data["train_images"]
    train_labels = [x.reshape(26, 1) for x in data["train_labels"]]
    test_images = data["test_images"]
    test_labels = [x.reshape(26, 1) for x in data["test_labels"]]
    return train_images, train_labels, test_images, test_labels

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases  = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        activations = [a]
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
            activations.append(a)
        return activations

    def evaluate(self, a):
        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    def Backpropagate(self, input, desired_output, learning_rate):
        activations = [input]
        a = input
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            a = sigmoid(z)
            activations.append(a)

        delta = (-2 * CalculateError(activations[-1], desired_output)) * sigmoid_prime(activations[-1])
        deltas = [delta]
        
        for l in range(2, self.num_layers):
            sp = sigmoid_prime(activations[-l])
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            deltas.insert(0, delta)
        
        for i in range(len(self.weights)):
            self.weights[i] += learning_rate * np.dot(deltas[i], activations[i].T)
            self.biases[i]  += learning_rate * deltas[i]

    def SaveData(self, path):
        for k, x in enumerate(self.weights):
            np.save(os.path.join(path, f"weights{k}.npy"), x)
        for k, x in enumerate(self.biases):
            np.save(os.path.join(path, f"biases{k}.npy"), x)
    
    def SaveImage(self, path):        
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
    
    for k, x in enumerate(TestingSet):
        desired = DesiredTSet[k]
        result = Net.evaluate(x)
        SumOfSum += np.sum((result - desired)**2)
        Total += 1

    return SumOfSum / Total

def ConvertDesiredIntoNeurons(Output):
    neurons = []
    for o in Output:
        match o:
            case 0:
                neurons.append(np.array([1,0,0,0,0,0,0,0,0,0]).reshape(10,1))
            case 1:
                neurons.append(np.array([0,1,0,0,0,0,0,0,0,0]).reshape(10,1))
            case 2:
                neurons.append(np.array([0,0,1,0,0,0,0,0,0,0]).reshape(10,1))
            case 3:
                neurons.append(np.array([0,0,0,1,0,0,0,0,0,0]).reshape(10,1))
            case 4:
                neurons.append(np.array([0,0,0,0,1,0,0,0,0,0]).reshape(10,1))
            case 5:
                neurons.append(np.array([0,0,0,0,0,1,0,0,0,0]).reshape(10,1))
            case 6:
                neurons.append(np.array([0,0,0,0,0,0,1,0,0,0]).reshape(10,1))
            case 7:
                neurons.append(np.array([0,0,0,0,0,0,0,1,0,0]).reshape(10,1))
            case 8:
                neurons.append(np.array([0,0,0,0,0,0,0,0,1,0]).reshape(10,1))
            case 9:
                neurons.append(np.array([0,0,0,0,0,0,0,0,0,1]).reshape(10,1))
    return neurons

net = Network([2500, 32, 32, 26])

Train, TrainL, Test, TestL = LoadDataset()

epochs = 30
accuracy = []

def exponential_decay(initial_lr, epoch, decay_rate):
    return initial_lr * np.exp(-decay_rate * epoch)

os.makedirs("Weights", exist_ok=True)
os.makedirs("VisualWeights", exist_ok=True)
os.makedirs("ProgressTracker", exist_ok=True)

initial_lr = 1.5
decay_rate = 0.15

for e in range(epochs):
    for x in tqdm(range(len(Train)), desc="Epoch Progress"):
        net.Backpropagate(Train[x], TrainL[x], exponential_decay(initial_lr, e, decay_rate))

    acc = CalculateCost(net, Test, TestL)
    accuracy.append(acc)
    print(f"Epoch {e+1}/{epochs} | Learning Rate: {exponential_decay(initial_lr, e, decay_rate)} | Cost: {acc}\n")

    net.SaveData("Weights")
    net.SaveImage("VisualWeights")

    plt.plot(range(1, len(accuracy) + 1), accuracy, linestyle='-', color='b', label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy %')
    plt.title('Accuracy Graph')
    plt.savefig(f"ProgressTracker\\{e}.png")
    np.save(f"ProgressTracker\\{e}.npy", np.array(accuracy))