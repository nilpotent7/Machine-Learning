import matplotlib.pyplot as plt
from tqdm import tqdm
from Models.LanguageModels import ProbabilisticNeuralModel

text = open("Train.txt", "r").read()
Model = ProbabilisticNeuralModel(ProbabilisticNeuralModel.WordBasedTokenizer, 8, 32, [], text)

def SplitData(Data, ratio):
    split_index = int(len(Data) * ratio)
    train_tokens = Data[:split_index]
    eval_tokens = Data[split_index:]
    return train_tokens, eval_tokens

def GetBatch(Data, BatchSize):
    for i in range(0, len(Data), BatchSize):
        yield Data[i:i + BatchSize]

Model.LoadData("Parameters")

loss = []
epochs = 100
Model.PrepareTraining(text, 2.5)
for e in tqdm(range(epochs), desc="Training"):
    Model.Train()
    loss.append(Model.LikelihoodLoss(text))

print(f"Final Loss: {Model.LikelihoodLoss(text):.6f}")

if(input("Save? (Y/N)\n").lower() == "y"): 
    Model.SaveData("Parameters")

plt.plot(loss, label="Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.savefig("Progress.png")
plt.close()