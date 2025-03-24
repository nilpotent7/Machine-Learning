import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from Models.NeuralNetwork import NeuralNetwork
from MNIST import MnistDataLoader

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

    return (SumOfSum / Total).item()

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
def ZScoreStandardize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1
    standardized_data = (data - mean) / std
    return standardized_data

# Perform Minimum Maximum Range Standardization on Input
def MinMaxStandardize(data):
    data_min = np.min(data)
    data_max = np.max(data)
    range_val = data_max - data_min
    if range_val == 0: return np.zeros_like(data)
    return (data - data_min) / range_val

MnistLoader = MnistDataLoader("Dataset")
Data = MnistLoader.LoadData()

Inputs = [MinMaxStandardize(np.array(x).reshape(784,1)) for x in Data[0][0]]
TestInputs = [MinMaxStandardize(np.array(x).reshape(784,1)) for x in Data[1][0]]

DesiredOutputs = EncodeOneHot(Data[0][1], 10)
TestDesiredOutputs = EncodeOneHot(Data[1][1], 10)


os.makedirs("Progress\\Raw\\Accuracy", exist_ok=True)
os.makedirs("Progress\\Raw\\Loss", exist_ok=True)
os.makedirs("Progress\\Accuracy", exist_ok=True)
os.makedirs("Progress\\Loss", exist_ok=True)

epochs = 25
lossProgress = []
accuracyProgress = []

net = NeuralNetwork([784, 64, 64, 10], NeuralNetwork.Sigmoid, NeuralNetwork.Softmax, NeuralNetwork.CrossEntropyLoss)
net.UseAdamW(LearningRate=0.001)
LossFunction = CrossEntropyLoss

for epoch in range(epochs):
    for i in tqdm(range(len(Inputs)), desc="Epoch Progress"):
        net.Train(Inputs[i], DesiredOutputs[i], epoch)

    loss = LossFunction(net, TestInputs, TestDesiredOutputs)
    lossProgress.append(loss)

    acc = CalculateAccuracy(net, TestInputs, TestDesiredOutputs)
    accuracyProgress.append(acc)
    
    print(f"Epoch {epoch+1}/{epochs} | Accuracy: {acc:.4f} | Loss: {loss:.4f} | LR: {net.CurrentLearningRate(epoch):.4f}\n")

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