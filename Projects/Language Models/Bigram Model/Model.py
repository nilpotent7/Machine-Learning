import numpy as np
from collections import Counter

class BigramModel(object):

    def __init__(self, Tokenizer, Text, ContextLimit):
        self.tokenizer = Tokenizer
        self.context_limit = ContextLimit

        self.vocab = list(set(self.tokenizer(Text)))
        self.vocab.sort()
        self.vocabSize = len(self.vocab)
        
        self.bigram = np.zeros((self.vocabSize, self.vocabSize))
        # self.bigram = np.random.rand(self.vocabSize, self.vocabSize)
        # row_sums = self.bigram.sum(axis=1, keepdims=True)
        # row_sums[row_sums == 0] = 1
        # self.bigram /= row_sums

        self.IndexMap = {word: i for i, word in enumerate(self.vocab)}
        self.WordMap = {i: word for word, i in self.IndexMap.items()}

        

    @staticmethod
    def Tokenize(Text):
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

            


# Example
text = 'In natural language processing, a word embedding is a representation of a word. The embedding is used in text analysis. Typically, the representation is a real-valued vector that encodes the meaning of the word in such a way that the words that are closer in the vector space are expected to be similar in meaning.'
Model = BigramModel(BigramModel.Tokenize, text, 250)
print(Model.Generate("In natural language", 25))

Model.Train(text)
print(Model.Generate("In natural language", 25))
