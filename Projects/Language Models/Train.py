import numpy as np
from tqdm import tqdm
from Models.Transformer import TransformerModel, WordBasedTokenizer

def GetRandomBatch(Data, ContextSize, BatchSize):
    ix = np.random.randint(0, len(Data) - ContextSize, size=(BatchSize,))
    x = np.stack([Data[i:i+ContextSize] for i in ix])
    y = np.stack([Data[i+1:i+ContextSize+1] for i in ix])
    return x,y

FullDataSet = "In natural language processing, a word embedding is a representation of a word. The embedding is used in text analysis. Typically, the representation is a real-valued vector that encodes the meaning of the word in such a way that the words that are closer in the vector space are expected to be similar in meaning."
tokenizer = WordBasedTokenizer(FullDataSet)

HeadSize = 16
EmbeddingSize = 8
Context = 4

Model = TransformerModel(tokenizer.GetLength(), Context, EmbeddingSize, HeadSize)
Model.LoadModel("Parameters")

FullDataSet = tokenizer.Tokenize(FullDataSet)

N = int(0.9*len(FullDataSet))
Train = FullDataSet[:N]
Test = FullDataSet[N:][:FullDataSet.index(0) + 1]

BatchSize = 4

Xs, Ys = GetRandomBatch(Train, Context, BatchSize)

Test = False
for epoch in tqdm(range(5000), desc="Training Progress"):
    for i in range(len(Ys)):
        Model.Train(Xs[i], Ys[i], LearningRate=0.1)

Model.SaveModel("Parameters")