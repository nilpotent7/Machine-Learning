import numpy as np
from tqdm import tqdm
from Models.LanguageModels import ProbabilisticNeuralModel

text = open("Train.txt", "r").read()
Model = ProbabilisticNeuralModel(ProbabilisticNeuralModel.WordBasedTokenizer, 6, 2, [128], text)

print(Model.Generate("In natural language", 25))

epochs = 500
Model.PrepareTraining(text, 0.1)
for e in tqdm(range(epochs), desc="Training"):
    Model.Train()

print(f"Final Loss: {Model.LikelihoodLoss(text):.4f}")
    

print(Model.Generate("In natural language", 25))
