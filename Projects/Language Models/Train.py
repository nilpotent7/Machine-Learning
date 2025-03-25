import matplotlib.pyplot as plt
from tqdm import tqdm
from Models.LanguageModels import ProbabilisticNeuralModel

text = open("Train.txt", "r").read()
Model = ProbabilisticNeuralModel(ProbabilisticNeuralModel.WordBasedTokenizer, 32, 4, [256], text)
print(Model.vocabSize)

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