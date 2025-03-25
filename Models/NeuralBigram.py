import numpy as np
from Models.NeuralNetwork import NeuralNetwork

class NeuralBigramModel(object):

    def __init__(self, Tokenizer, Text, LearningRate, Layers):
        self.tokenizer = Tokenizer

        self.vocab = list(set(self.tokenizer(Text)))
        self.vocab.sort()
        self.vocabSize = len(self.vocab)
        
        self.network = NeuralNetwork([self.vocabSize, *Layers, self.vocabSize], NeuralNetwork.Sigmoid, NeuralNetwork.Softmax, NeuralNetwork.CrossEntropyLoss)
        self.network.UseSGD(LearningRate)

        self.IndexMap = {word: i for i, word in enumerate(self.vocab)}
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
            self.network.Train(self.OneHotEncode(self.IndexMap[x]), self.OneHotEncode(self.IndexMap[y]), 10)