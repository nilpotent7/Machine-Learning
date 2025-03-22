import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from Model import NeuralNetwork

def CalculateAccuracy(Network, Test, Desired):
    correct = 0
    total = 0
    
    for x, desired in zip(Test, Desired):
        output = Network.Evaluate(x)
        
        predicted = np.argmax(output)
        true_label = np.argmax(desired)
        
        if predicted == true_label: correct += 1
        total += 1
        
    return correct / total

# Entire Network's Cost Function: Mean Squared Error
def MeanSquaredError(Network, Test, Desired):
    SumOfSum = 0
    Total = 0
    
    for k,x in enumerate(Test):
        desired = Desired[k]
        result = Network.Evaluate(x)

        overall_sum = 0
        for x,y in zip(result, desired):
            overall_sum += ((x-y)*(x-y))
        
        SumOfSum += overall_sum
        Total += 1

    return SumOfSum / Total

# Entire Network's Cost Function: Cross Entropy Loss
def CrossEntropyLoss(Network, Test, Desired):
    total_cost = 0.0
    total = 0
    epsilon = 1e-12
    
    for k, x in enumerate(Test):
        desired = Desired[k]
        result = Network.Evaluate(x)
        
        cost = -np.sum(desired * np.log(result + epsilon))
        total_cost += cost
        total += 1
        
    return total_cost / total

# Encode Desired Output into One-Hot neurons
def EncodeOneHot(Output, Length):
    outputs = np.array(Output)
    neurons = np.eye(Length)[outputs]
    return neurons.reshape(outputs.size, Length, 1)

# Perform Z-Score Standardization on Input
def Standardize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1
    standardized_data = (data - mean) / std
    return standardized_data

def LoadDataset(path="dataset.npz"):
    data = np.load(path)
    train_images = data["train_images"]
    train_labels = [x.reshape(26, 1) for x in data["train_labels"]]
    test_images = data["test_images"]
    test_labels = [x.reshape(26, 1) for x in data["test_labels"]]
    return train_images, train_labels, test_images, test_labels

Train, TrainL, Test, TestL = LoadDataset()

epochs = 25
lossProgress = []
accuracyProgress = []

net = NeuralNetwork([2500, 32, 32, 26], NeuralNetwork.Sigmoid, NeuralNetwork.Softmax, NeuralNetwork.CrossEntropyLoss)
net.UseSGD(LearningRate=1.5, DecayFunction=NeuralNetwork.ExponentialDecay, DecayRate=0.15)
LossFunction = CrossEntropyLoss

os.makedirs("Weights", exist_ok=True)
os.makedirs("VisualWeights", exist_ok=True)
os.makedirs("ProgressTracker", exist_ok=True)

for epoch in range(epochs):
    for x in tqdm(range(len(Train)), desc="Epoch Progress"):
        net.Train(Train[x], TrainL[x], epoch)

    loss = LossFunction(net, Test[x], TestL[x])
    lossProgress.append(loss)

    acc = CalculateAccuracy(net, Test[x], TestL[x])
    accuracyProgress.append(acc)
    
    print(f"Epoch {epoch+1}/{epochs} | Accuracy: {acc} | Loss: {loss}\n")

    plt.figure()
    plt.plot(range(1, len(lossProgress) + 1), lossProgress, linestyle='-', color='b', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epoch')
    plt.savefig(f"Progress\\Loss\\{epoch}.png")
    np.save(f"Progress\\Raw\\Loss\\{epoch}.npy", np.array(lossProgress))
    plt.close()

    plt.figure()
    plt.plot(range(1, len(accuracyProgress) + 1), accuracyProgress, linestyle='-', color='b', label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epoch')
    plt.savefig(f"Progress\\Accuracy\\{epoch}.png")
    np.save(f"Progress\\Raw\\Accuracy\\{epoch}.npy", np.array(accuracyProgress))
    plt.close()

net.SaveData("Weights")
net.SaveDataAsImage("VisualWeights")