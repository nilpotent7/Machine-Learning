from Models.LanguageModels import BigramModel

text = open("Train.txt", "r").read()
Model = BigramModel(BigramModel.WordBasedTokenizer, text)

print(Model.Generate("In natural language", 25))
Model.Train(text)
print(Model.Generate("In natural language", 25))
