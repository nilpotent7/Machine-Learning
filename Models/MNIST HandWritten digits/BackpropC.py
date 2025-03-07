import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
import struct
from array import array

def z_score_standardize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1
    standardized_data = (data - mean) / std
    return standardized_data

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

class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img
        
        return images, labels
        
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)

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

net = Network([784, 64, 64, 10])

MnistLoader = MnistDataloader("Dataset\\train-images.idx3-ubyte",
                              "Dataset\\train-labels.idx1-ubyte",
                              "Dataset\\t10k-images.idx3-ubyte",
                              "Dataset\\t10k-labels.idx1-ubyte")
Data = MnistLoader.load_data()
Inputs = [z_score_standardize(np.array(x).reshape(784,1)) for x in Data[0][0]]
TestInputs = [z_score_standardize(np.array(x).reshape(784,1)) for x in Data[1][0]]
DesiredOutputs = ConvertDesiredIntoNeurons(Data[0][1])
TestDesiredOutputs = ConvertDesiredIntoNeurons(Data[1][1])

epochs = 100
accuracy = []

for e in range(epochs):
    for x in tqdm(range(len(Inputs)), desc="Epoch Progress"):
        net.Backpropagate(Inputs[x], DesiredOutputs[x], 0.01)

    acc = CalculateCost(net, TestInputs, TestDesiredOutputs)
    accuracy.append(acc)
    print(f"Epoch {e+1}/{epochs} Cost: {acc}\n")

    net.SaveData("Weights")
    net.SaveImage("VisualWeights")

    plt.plot(range(1, len(accuracy) + 1), accuracy, linestyle='-', color='b', label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy %')
    plt.title('Accuracy Graph')
    plt.savefig(f"ProgressTracker\\{e}.png")
    np.save(f"ProgressTracker\\{e}.npy", np.array(accuracy))