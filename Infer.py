import os
import struct
import numpy as np

from PIL import Image
from array import array

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
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
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
        if rounded: o = np.round(a, decimals=4) * 100
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
        print()

def CalculateCost(Net, TestingSet, DesiredTSet):
    SumOfSum = 0
    Total = 0
    
    for k,x in enumerate(TestingSet):
        desired = DesiredTSet[k]
        result = Net.evaluate(x, False)
        
        overall_sum = 0
        for x,y in zip(result, desired):
            overall_sum += ((x-y)*(x-y))
        
        SumOfSum += overall_sum
        Total +=1

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

def LoadImageInput(Path):
    img = Image.open(Path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img, dtype=np.float64)
    img_array = img_array.reshape((28*28, 1)) / 255
    return img_array

def PrintImage(Input, SavePath):
    img_array = Input.reshape((28, 28))
    img_array = img_array.astype(np.uint8)
    img = Image.fromarray(img_array, mode='L')
    img.save(SavePath)

# c = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# MnistLoader = MnistDataloader("Dataset\\train-images.idx3-ubyte",
#                               "Dataset\\train-labels.idx1-ubyte",
#                               "Dataset\\t10k-images.idx3-ubyte",
#                               "Dataset\\t10k-labels.idx1-ubyte")
# Data = MnistLoader.load_data()
# Inputs = [np.array(x).reshape(784,1) / 255 for x in Data[1][0]]
# DesiredOutputs = ConvertDesiredIntoNeurons(Data[1][1])

Input = LoadImageInput("Input.png")

net = Network([2, 64, 64, 1 ])
net.LoadData("Models\\Misc\\Data")

# print("Model Error: " + str(CalculateCost(net, Inputs, DesiredOutputs)))
# OutputResult(, c)
print(net.evaluate(Input, True))