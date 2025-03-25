from Models.LanguageModels import ProbabilisticNeuralModel

Model = ProbabilisticNeuralModel(ProbabilisticNeuralModel.WordBasedTokenizer, 32, 4, [256])
Model.LoadData("Parameters")

print(Model.Generate("Breaking fundamental components of NLP such as deep learning and language-based models, allows nuances of spoken language to be", 2500))
