import numpy as np

def Softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def CrossEntropy (result, desired):
    epsilon = 1e-12
    return -np.sum(desired * np.log(result + epsilon))

def CrossEntropyDerivative(result, desired):
    return result - desired

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


class LinearHead(object):
    def __init__(self, InSize, OutSize):
        self.weights = np.random.randn(InSize, OutSize) * (InSize*OutSize)**-0.5
        
    def Forward(self, X):
        return np.dot(X, self.weights)

class AttentionHead(object):

    def __init__(self, HeadSize, EmbeddingSize, ContextSize):
        self.embedSize = EmbeddingSize
        self.key = LinearHead(EmbeddingSize, HeadSize)
        self.query = LinearHead(EmbeddingSize, HeadSize)
        self.value = LinearHead(EmbeddingSize, HeadSize)
        self.tril = np.tril(np.ones((ContextSize, ContextSize)))
    
    def Forward(self, X):
        T = len(X)
        k = self.key.Forward(X)
        q = self.query.Forward(X)

        Wi = np.dot(q, k.T) * self.embedSize**-0.5
        Wi = np.where(self.tril[:T, :T] == 0, float("-inf"), Wi)
        Wi = Softmax(Wi)

        v = self.value.Forward(X)
        out = np.dot(Wi, v)
        return out

class TransformerModel(object):

    def __init__(self, Tokenizer, ContextSize, EmbeddingSize, HeadSize, Text=""):
        self.tokenizer = Tokenizer
        self.embeddingSize = EmbeddingSize
        self.contextSize = ContextSize

        vocab = list(set(self.tokenizer(Text)[:-1]))
        vocab.sort()
        vocab.insert(0, '<S>')
        self.vocabSize = len(vocab)


        self.token_embedding = np.random.randn(self.vocabSize, EmbeddingSize)
        self.position_embedding = np.random.randn(ContextSize, EmbeddingSize)

        self.attentionHead = AttentionHead(HeadSize, EmbeddingSize, ContextSize)
        self.languageHead = LinearHead(HeadSize, self.vocabSize)


        self.IndexMap = { word: idx for idx, word in enumerate(vocab) }
        self.WordMap = { idx: word for idx, word in enumerate(vocab) }
        
    def Predict(self, idx):
        T = len(idx)
        tok_embed = self.token_embedding[idx]
        pos_embed = self.position_embedding[np.arange(T)]
        x = tok_embed + pos_embed
        x = self.attentionHead.Forward(x)
        logits = self.languageHead.Forward(x)
        return logits

    def Generate(self, Start, MaxTokens):
        sentence = Start
        idx = self.tokenizer(Start)[:-1]
        idx = [self.IndexMap[x] for x in idx]

        for i in range(MaxTokens):
            context = idx[-self.contextSize:]
            logits = self.Predict(context)[-1:].T
            probs = Softmax(logits)
            pred = np.random.choice(self.vocabSize, p=probs.reshape(self.vocabSize))
            context = context[1:] + [pred]
            sentence += self.WordMap[pred]

        return sentence
            

text = "In natural language processing, a word embedding is a representation of a word. The embedding is used in text analysis. Typically, the representation is a real-valued vector that encodes the meaning of the word in such a way that the words that are closer in the vector space are expected to be similar in meaning."
Model = TransformerModel(WordBasedTokenizer, 4, 8, 8, text)
print(Model.Generate("In natural language", 25))
