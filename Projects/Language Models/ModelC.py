import numpy as np
from tqdm import tqdm
from Models.LanguageModels import ProbabilisticNeuralModel

text = open("Train.txt", "r").read()
Model = ProbabilisticNeuralModel(ProbabilisticNeuralModel.WordBasedTokenizer, text, 6, 8, [128])

print(Model.Generate("natural language is a vector"), 2500)

epochs = 500
batchSize = 36
Model.PrepareTraining(text, batchSize, 0.1)
for e in tqdm(range(epochs), desc="Training"):
    Model.Train()

print(f"Final Loss: {Model.LikelihoodLoss(text):.4f}")
    

print(Model.Generate("natural language is a vector"), 2500)
