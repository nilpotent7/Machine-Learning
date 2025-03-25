import os
import numpy as np
from Models.NeuralNetwork import NeuralNetwork

class BigramModel(object):

    def __init__(self, Tokenizer, Text):
        self.tokenizer = Tokenizer

        self.vocab = list(set(self.tokenizer(Text)))
        self.vocab.sort()
        self.vocabSize = len(self.vocab)
        
        self.bigram = np.zeros((self.vocabSize, self.vocabSize))

        self.IndexMap = { word: idx for idx, word in enumerate(self.vocab) }
        self.WordMap = { idx: word for idx, word in enumerate(self.vocab) }

    def LikelihoodLoss(self, Text):
        log_likelihood = 0
        for x,y in zip(Text, Text[1:]):
            prob = self.bigram[self.IndexMap[x], self.IndexMap[y]]
            log_likelihood += np.log(prob)
        return -log_likelihood / (len(Text)-1)
        

    @staticmethod
    def WordBasedTokenizer(Text):
        tokens = []
        word = ""
        i = 0

        while i < len(Text):
            char = Text[i]

            if char.isalnum(): word += char
            elif char in ("'", "’") and word: word += char

            else:
                if word:
                    tokens.append(word)
                    word = ""

                if char == " ": tokens.append(" ")
                elif char == "-": tokens.append("-")
                elif char.strip(): tokens.append(char)
            i += 1

        if word: tokens.append(word)

        return tokens
        
    def Generate(self, start, max_tokens):
        sentence = self.tokenizer(start)
        current_word = sentence[-1]

        for _ in range(max_tokens - 1):
            next_word_probs = self.bigram[self.IndexMap[current_word]]
            if next_word_probs.sum() == 0:
                next_word = np.random.choice(self.vocab)
                print
            else:
                next_word_index = np.random.choice(self.vocabSize, p=next_word_probs)
                next_word = self.WordMap[next_word_index]
            sentence.append(next_word)
            current_word = next_word

        return ''.join(sentence)
    
    def Train(self, Text):
        Text = self.tokenizer(Text)
        for x,y in zip(Text, Text[1:]):
            self.bigram[self.IndexMap[x], self.IndexMap[y]] += 1
        
        row_sums = self.bigram.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.bigram /= row_sums

class NeuralBigramModel(object):

    def __init__(self, Tokenizer, LearningRate, Layers, Text=""):
        self.tokenizer = Tokenizer

        vocab = list(set(self.tokenizer(Text)))
        vocab.sort()
        self.vocabSize = len(vocab)
        
        self.network = NeuralNetwork([self.vocabSize, *Layers, self.vocabSize], NeuralNetwork.Sigmoid, NeuralNetwork.Softmax, NeuralNetwork.CrossEntropyLoss)
        self.network.UseSGD(LearningRate)

        self.IndexMap = {word: i for i, word in enumerate(vocab)}
        self.WordMap = {i: word for word, i in self.IndexMap.items()}

    
    def OneHotEncode(self, data):
        return np.eye(self.vocabSize)[data].reshape(self.vocabSize, 1)
    
    def LikelihoodLoss(self, Text):
        Text = self.tokenizer(Text)
        log_likelihood = 0
        for x,y in zip(Text, Text[1:]):
            prob = self.network.Evaluate(self.OneHotEncode(self.IndexMap[x]))[self.IndexMap[y]]
            log_likelihood += np.log(prob)
        return (-log_likelihood / (len(Text)-1)).item()

    @staticmethod
    def WordBasedTokenizer(Text):
        tokens = []
        word = ""
        i = 0

        while i < len(Text):
            char = Text[i]

            if char.isalnum(): word += char
            elif char in ("'", "’") and word: word += char

            else:
                if word:
                    tokens.append(word)
                    word = ""

                if char == " ": tokens.append(" ")
                elif char == "-": tokens.append("-")
                elif char.strip(): tokens.append(char)
            i += 1

        if word: tokens.append(word)

        return tokens
        
    def Generate(self, start, max_tokens):
        sentence = self.tokenizer(start)
        current_word = sentence[-1]

        for _ in range(max_tokens - 1):
            probs = self.network.Evaluate(self.OneHotEncode(self.IndexMap[current_word])).reshape(self.vocabSize)
            index = np.random.choice(self.vocabSize, p=probs)
            word = self.WordMap[index]
            sentence.append(word)
            current_word = word

        return ''.join(sentence)
    
    def Train(self, Text):
        Text = self.tokenizer(Text)
        for x,y in zip(Text, Text[1:]):
            x = self.OneHotEncode(self.IndexMap[x])
            y = self.OneHotEncode(self.IndexMap[y])
            self.network.Train(x, y, 10)
    
    def SaveData(self, path, filename="model.npz"):
        self.network.SaveData(path)
        np.save(os.path.join(path, filename), IndexMap=self.IndexMap, WordMap=self.WordMap, VocabSize=self.vocabSize)
        
    def LoadData(self, path, filename="model.npz"):
        self.network.LoadData(path)
        data = np.load(os.path.join(path, filename), allow_pickle=True)
        self.IndexMap = data["IndexMap"].item()
        self.WordMap = data["WordMap"].item()
        self.vocabSize = data["VocabSize"].item()

class ProbabilisticNeuralModel(object):

    def __init__(self, Tokenizer, ContextSize, EmbeddingSize, Layers, Text=""):
        self.tokenizer = Tokenizer
        self.embeddingSize = EmbeddingSize
        self.contextSize = ContextSize

        vocab = list(set(self.tokenizer(Text)[:-1]))
        vocab.sort()
        vocab.insert(0, '<S>')
        self.vocabSize = len(vocab)

        self.embedding = np.random.randn(self.vocabSize, EmbeddingSize)
        self.IndexMap = { word: idx for idx, word in enumerate(vocab) }
        self.WordMap = { idx: word for idx, word in enumerate(vocab) }
        
        self.network = NeuralNetwork(
            [self.embeddingSize*self.contextSize, *Layers, self.vocabSize],
            NeuralNetwork.Tanh,
            NeuralNetwork.Softmax,
            NeuralNetwork.CrossEntropyLoss
        )

    
    def OneHotEncode(self, data):
        return np.eye(self.vocabSize)[data].reshape(self.vocabSize, 1)
    
    def LikelihoodLoss(self, Text):
        Text = self.tokenizer(Text)
        log_likelihood = 0
        Context = ([0]*self.contextSize + [self.IndexMap[t] for t in Text] + [0]*self.contextSize)[-self.contextSize:]

        for token in Text:
            ix = self.IndexMap[token]
            x = self.embedding[Context]
            x = x.reshape(x.shape[0]*x.shape[1], 1)
            Context = Context[1:] + [ix]

            prob = self.network.Evaluate(x)[self.IndexMap[token]]
            log_likelihood += np.log(prob)
        return (-log_likelihood / (len(Text)-1)).item()

    @staticmethod
    def WordBasedTokenizer(Text):
        tokens = []
        word = ""
        i = 0

        while i < len(Text):
            char = Text[i]

            if char.isalnum(): word += char
            elif char in ("'", "’") and word: word += char

            else:
                if word:
                    tokens.append(word)
                    word = ""

                if char == " ": tokens.append(" ")
                elif char == "-": tokens.append("-")
                elif char.strip(): tokens.append(char)
            i += 1

        if word: tokens.append(word)

        return tokens + ['<S>']
        
    def Generate(self, start, max_tokens):
        sentence = self.tokenizer(start)[:-1]
        Context = (['<S>']*self.contextSize + sentence)[-self.contextSize:]
        Context = [self.IndexMap[x] for x in Context]

        for _ in range(max_tokens - 1):
            X = self.embedding[Context]
            X = X.reshape((X.shape[0] * X.shape[1]), 1)
            
            probs = self.network.Evaluate(X).reshape(self.vocabSize)
            index = np.random.choice(self.vocabSize, p=probs)
            Context = Context[1:] + [index]

            word = self.WordMap[index]
            sentence.append(word)
            if(index == 0): break

        return ''.join(sentence)
    
    def PrepareTraining(self, Text, LearningRate):
        self.network.UseSGD(LearningRate)
        self.embeddingLR = LearningRate

        Text = self.tokenizer(Text)
        Context = [0] * self.contextSize
        self.X = []
        self.Y = []
        for token in Text:
            ix = self.IndexMap[token]
            self.X.append(Context)
            self.Y.append(ix)
            Context = Context[1:] + [ix]

        # Desired Outputs
        self.Y = np.array(self.Y).reshape(len(Text), 1)

        self.network.UseSGD

    def Train(self):
        for x,y in zip(self.X, self.Y):
            x = np.array(x)
            X = self.embedding[np.array(x)]
            X = X.reshape((X.shape[0] * X.shape[1], 1))
            Y = self.OneHotEncode(y.reshape(y.shape[0], 1))

            grad = self.network.Train(X, Y).reshape(self.contextSize, self.embeddingSize)

            # Update each embedding vector corresponding to the context tokens
            for pos, idx in enumerate(x):
                self.embedding[idx] -= self.embeddingLR * grad[pos]

    def SaveData(self, path, filename="model.npz"):
        self.network.SaveData(path)
        np.savez_compressed(os.path.join(path, filename), Embedding=self.embedding, IndexMap=self.IndexMap, WordMap=self.WordMap, VocabSize=self.vocabSize)
        
    def LoadData(self, path, filename="model.npz"):
        self.network.LoadData(path)
        data = np.load(os.path.join(path, filename), allow_pickle=True)
        self.embedding = data["Embedding"]
        self.IndexMap = data["IndexMap"].item()
        self.WordMap = data["WordMap"].item()
        self.vocabSize = data["VocabSize"].item()

class TransformerModel(object):

    def __init__(self, Tokenizer, ContextSize, EmbeddingSize, Layers, Text=""):
        self.tokenizer = Tokenizer
        self.embeddingSize = EmbeddingSize
        self.contextSize = ContextSize

        vocab = list(set(self.tokenizer(Text)[:-1]))
        vocab.sort()
        vocab.insert(0, '<S>')
        self.vocabSize = len(vocab)

        self.position_embedding = np.random.randn(ContextSize, EmbeddingSize)
        self.embedding = np.random.randn(self.vocabSize, EmbeddingSize)

        self.IndexMap = { word: idx for idx, word in enumerate(vocab) }
        self.WordMap = { idx: word for idx, word in enumerate(vocab) }
        
        self.network = NeuralNetwork(
            [self.embeddingSize, *Layers, self.vocabSize],
            NeuralNetwork.Linear,
            NeuralNetwork.Linear,
            NeuralNetwork.CrossEntropyLoss
        )

    
    def OneHotEncode(self, data):
        return np.eye(self.vocabSize)[data].reshape(self.vocabSize, 1)
    
    def LikelihoodLoss(self, Text):
        Text = self.tokenizer(Text)
        log_likelihood = 0
        Context = ([0]*self.contextSize + [self.IndexMap[t] for t in Text] + [0]*self.contextSize)[-self.contextSize:]

        for token in Text:
            ix = self.IndexMap[token]
            x = self.embedding[Context]
            x = x.reshape(x.shape[0]*x.shape[1], 1)
            Context = Context[1:] + [ix]

            prob = self.network.Evaluate(x)[self.IndexMap[token]]
            log_likelihood += np.log(prob)
        return (-log_likelihood / (len(Text)-1)).item()

    @staticmethod
    def WordBasedTokenizer(Text):
        tokens = []
        word = ""
        i = 0

        while i < len(Text):
            char = Text[i]

            if char.isalnum(): word += char
            elif char in ("'", "’") and word: word += char

            else:
                if word:
                    tokens.append(word)
                    word = ""
                    
                if char == "\n": tokens.append("\n")
                elif char == " ": tokens.append(" ")
                elif char == "-": tokens.append("-")
                elif char.strip(): tokens.append(char)
            i += 1

        if word: tokens.append(word)

        return tokens + ['<S>']
        
    def Generate(self, start, max_tokens):
        sentence = self.tokenizer(start)[:-1]
        Context = (['<S>']*self.contextSize + sentence)[-self.contextSize:]
        Context = [self.IndexMap[x] for x in Context]

        for _ in range(max_tokens - 1):
            X = self.embedding[Context]
            X = X.reshape((X.shape[0] * X.shape[1]), 1)
            
            probs = self.network.Evaluate(X).reshape(self.vocabSize)
            index = np.random.choice(self.vocabSize, p=probs)
            Context = Context[1:] + [index]

            word = self.WordMap[index]
            sentence.append(word)
            if(index == 0): break

        return ''.join(sentence)
    
    def PrepareNeuralTraining(self, LearningRate):
        self.network.UseSGD(LearningRate)
        self.embeddingLR = LearningRate
    
    def PrepareTraining(self, Text, LearningRate):
        self.PrepareNeuralTraining()

        Text = self.tokenizer(Text)
        Context = [0] * self.contextSize
        self.X = []
        self.Y = []
        for token in Text:
            ix = self.IndexMap[token]
            self.X.append(Context)
            self.Y.append(ix)
            Context = Context[1:] + [ix]

        # Desired Outputs
        self.Y = np.array(self.Y).reshape(len(Text), 1)


    def Train(self):
        for x,y in zip(self.X, self.Y):
            x = np.array(x)
            X = self.embedding[np.array(x)]
            X = X.reshape((X.shape[0] * X.shape[1], 1))
            Y = self.OneHotEncode(y.reshape(y.shape[0], 1))

            grad = self.network.Train(X, Y).reshape(self.contextSize, self.embeddingSize)

            # Update each embedding vector corresponding to the context tokens
            for pos, idx in enumerate(x):
                self.embedding[idx] -= self.embeddingLR * grad[pos]

    def TrainBatch(self, Batch):
        # Batch.shape = [T]
        # X.shape = [T,C]
        # logits = [T,vocabSize]
        T = len(Batch)
        X = self.embedding[Batch] + self.position_embedding[np.arange(T)]
        logits = []

        for i in range(T):
            logits.append(self.network.Evaluate(X[i]))

    def SaveData(self, path, filename="model.npz"):
        self.network.SaveData(path)
        np.savez_compressed(os.path.join(path, filename), Embedding=self.embedding, IndexMap=self.IndexMap, WordMap=self.WordMap, VocabSize=self.vocabSize)
        
    def LoadData(self, path, filename="model.npz"):
        self.network.LoadData(path)
        data = np.load(os.path.join(path, filename), allow_pickle=True)
        self.embedding = data["Embedding"]
        self.IndexMap = data["IndexMap"].item()
        self.WordMap = data["WordMap"].item()
        self.vocabSize = data["VocabSize"].item()

