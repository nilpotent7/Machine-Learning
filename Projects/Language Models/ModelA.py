from Models.LanguageModels import BigramModel

text = open("Train.txt", "r").read()
Model = BigramModel(BigramModel.WordBasedTokenizer, text)

print(Model.Generate("when there is a temperature", 25))
Model.Train(text)
print(Model.Generate("when there is a temperature", 25))
