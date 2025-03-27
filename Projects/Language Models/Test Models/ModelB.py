from tqdm import tqdm
from Models.LanguageModels import NeuralBigramModel

text = open("Train.txt", "r").read()
Model = NeuralBigramModel(NeuralBigramModel.WordBasedTokenizer, text, 1, [])

print(Model.Generate("In natural language", 25))

epochs = 1000
for e in tqdm(range(epochs), desc="Training"):
    Model.Train(text)

print(f"Final Loss: {Model.LikelihoodLoss(text):.4f}")
    

print(Model.Generate("In natural language", 25))
