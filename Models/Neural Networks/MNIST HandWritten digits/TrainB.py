import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import struct
from array import array
import os

np.random.seed(51)

def FindFiles(directory_path):
    file_paths = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(a):
    return a * (1 - a)

class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    def evaluate(self, a):
        return self.feedforward(a)
    
    def Backpropagate(self, batch_inputs, batch_outputs, learning_rate):
        batch_size = len(batch_inputs)
        
        # Initialize gradients
        nabla_b = [np.zeros_like(b) for b in self.biases]
        nabla_w = [np.zeros_like(w) for w in self.weights]
        
        for input, desired_output in zip(batch_inputs, batch_outputs):
            activations = [input]
            zs = []
            
            # Forward pass
            a = input
            for b, w in zip(self.biases, self.weights):
                z = np.dot(w, a) + b
                zs.append(z)
                a = sigmoid(z)
                activations.append(a)
            
            # Backward pass
            delta = (-2 * (activations[-1] - desired_output)) * sigmoid_prime(activations[-1])
            nabla_b[-1] += delta
            nabla_w[-1] += np.dot(delta, activations[-2].T)
            
            for l in range(2, self.num_layers):
                delta = np.dot(self.weights[-l + 1].T, delta) * sigmoid_prime(activations[-l])
                nabla_b[-l] += delta
                nabla_w[-l] += np.dot(delta, activations[-l - 1].T)
        
        # Update weights and biases
        self.weights = [w + (learning_rate / batch_size) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b + (learning_rate / batch_size) * nb for b, nb in zip(self.biases, nabla_b)]

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

# Helper functions
def CalculateCost(Net, TestingSet, DesiredTSet):
    total_cost = 0
    for x, y in zip(TestingSet, DesiredTSet):
        result = Net.evaluate(x)
        total_cost += np.sum((result - y) ** 2)
    return total_cost / len(TestingSet)

def ConvertIntoHotShot(Output):
    return [np.eye(10)[o].reshape(10, 1) for o in Output]

MnistLoader = MnistDataloader("Dataset\\train-images.idx3-ubyte",
                              "Dataset\\train-labels.idx1-ubyte",
                              "Dataset\\t10k-images.idx3-ubyte",
                              "Dataset\\t10k-labels.idx1-ubyte")
Data = MnistLoader.load_data()
Inputs = [np.array(x).reshape(784, 1) for x in Data[0][0]]
TestInputs = [np.array(x).reshape(784, 1) for x in Data[1][0]]
DesiredOutputs = ConvertIntoHotShot(Data[0][1])
TestDesiredOutputs = ConvertIntoHotShot(Data[1][1])

# Training with Mini-Batch Gradient Descent
net = Network([784, 32, 32, 10])

batch_size = 32
epochs = 25
learning_rate = 0.01
accuracy = []

os.makedirs('ProgressTracker', exist_ok=True)
os.makedirs('ProgressTracker\\Data', exist_ok=True)
os.makedirs("Weights", exist_ok=True)
os.makedirs("VisualWeights", exist_ok=True)

acc = CalculateCost(net, TestInputs, TestDesiredOutputs)
accuracy.append(acc)
print(f"Initial Cost: {acc}\n")

for e in range(epochs):
    shuffled_indices = np.random.permutation(len(Inputs))
    Inputs = [Inputs[i] for i in shuffled_indices]
    DesiredOutputs = [DesiredOutputs[i] for i in shuffled_indices]
    
    for i in tqdm(range(0, len(Inputs), batch_size), desc=f"Epoch {e+1}/{epochs}"):
        batch_inputs = Inputs[i:i+batch_size]
        batch_outputs = DesiredOutputs[i:i+batch_size]
        net.Backpropagate(batch_inputs, batch_outputs, learning_rate)
    
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
    np.save(f"ProgressTracker\\Data\\{e}.npy", np.array(accuracy))
