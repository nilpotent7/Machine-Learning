import numpy as np

class BigramModel(object):

    def __init__(self, Tokenizer, Text):
        self.tokenizer = Tokenizer

        self.vocab = list(set(self.tokenizer(Text)))
        self.vocab.sort()
        self.vocabSize = len(self.vocab)
        
        self.bigram = np.zeros((self.vocabSize, self.vocabSize))

        self.IndexMap = {word: i for i, word in enumerate(self.vocab)}
        self.WordMap = {i: word for word, i in self.IndexMap.items()}

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