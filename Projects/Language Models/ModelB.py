from tqdm import tqdm
from Models.NeuralBigram import NeuralBigramModel

text = open("Train.txt", "r").read()
Model = NeuralBigramModel(NeuralBigramModel.WordBasedTokenizer, text, 10, [])

print(Model.Generate("In natural language", 25))

epochs = 250
for e in tqdm(range(epochs), desc="Training"):
    Model.Train(text)

print(f"Final Loss: {Model.LikelihoodLoss(text):.4f}")
    

print(Model.Generate("In natural language", 25))
