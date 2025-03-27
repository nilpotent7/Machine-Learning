import os
import numpy as np

def Softmax(z, axis=0):
    exp_z = np.exp(z - np.max(z, axis=axis, keepdims=True))
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)

# def CrossEntropy(result, target):
#     epsilon = 1e-12
#     N = result.shape[0]
#     return -np.sum(np.log(result[np.arange(N), target] + epsilon)) / N

# def CrossEntropyDerivative(result, target):
#     grad = result.copy()
#     grad[np.arange(result.shape[0]), target] -= 1
#     return grad / result.shape[0]

# def CrossEntropy(result, target):
#     epsilon = 1e-12
#     return -np.sum(target * np.log(result + epsilon))

# def CrossEntropyDerivative(result, target):
#     return result - target

def CrossEntropy(result, target):
    epsilon = 1e-12
    # Use advanced indexing to pick the probability of the target class for each sample
    return -np.sum(np.log(result[np.arange(result.shape[0]), target] + epsilon))

def CrossEntropyDerivative(result, target):
    grad = result.copy()
    grad[np.arange(result.shape[0]), target] -= 1
    return grad

def WordBasedSplit(Text):
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

class WordBasedTokenizer(object):
    def __init__(self, FullText):
        self.vocab = list(set(WordBasedSplit(FullText)[:-1]))
        self.vocab.sort()
        self.vocab.insert(0, '<S>')
        self.size = len(self.vocab)

        self.IndexMap = { word: idx for idx, word in enumerate(self.vocab) }
        self.WordMap = { idx: word for idx, word in enumerate(self.vocab) }

    def GetLength(self):
        return self.size

    def Tokenize(self, Text):
        return [self.IndexMap[x] for x in WordBasedSplit(Text)]
    
    def Untokenize(self, Tokens):
        return [self.WordMap[x] for x in Tokens]

class LinearHead(object):
    def __init__(self, InSize, OutSize):
        self.weights = np.random.randn(InSize, OutSize) / np.sqrt(InSize)
        self.biases = np.zeros(OutSize)

    # Perform Computation (Saving intermediatory steps for backpropogation)
    def Forward(self, X):
        self.last_input = X.copy()
        return np.dot(X, self.weights) + self.biases
    
    # Calculate Gradient
    def Backward(self, grad_output):
        self.grad_weights = np.dot(self.last_input.T, grad_output)
        self.grad_bias = np.sum(grad_output, axis=0)
        grad = np.dot(grad_output, self.weights.T)
        return grad

class AttentionHead(object):

    def __init__(self, HeadSize, EmbeddingSize, ContextSize):
        self.key = LinearHead(EmbeddingSize, HeadSize)
        self.query = LinearHead(EmbeddingSize, HeadSize)
        self.value = LinearHead(EmbeddingSize, HeadSize)
        self.tril = np.tril(np.ones((ContextSize, ContextSize)))

    def SetData(self, Data):
        self.key = Data[0].item()
        self.query = Data[1].item()
        self.value = Data[2].item()
    
    # Perform Computation (Saving intermediatory steps for backpropogation)
    def Forward(self, X):
        T = len(X)
        self.k = self.key.Forward(X)
        self.q = self.query.Forward(X)
        
        d_k = self.q.shape[-1]
        Wi = np.dot(self.q, self.k.T) / np.sqrt(d_k)
        
        self.mask = (self.tril[:T, :T] == 0)
        Wi[self.mask] = -1e9
        self.a = Softmax(Wi, axis=1)
        
        self.v = self.value.Forward(X)
        return np.dot(self.a, self.v)
    
    # Calculate Gradient
    def Backward(self, grad_output):
        grad_a = np.dot(grad_output, self.v.T)
        grad_v = np.dot(self.a.T, grad_output)

        grad_Wi = self.a * (grad_a - np.sum(grad_a * self.a, axis=1, keepdims=True))
        grad_Wi[self.mask] = 0

        grad_q = np.dot(grad_Wi, self.k)
        grad_k = np.dot(grad_Wi.T, self.q)

        grad_from_key = self.key.Backward(grad_k)
        grad_from_query = self.query.Backward(grad_q)
        grad_from_value = self.value.Backward(grad_v)

        grad_X = grad_from_key + grad_from_query + grad_from_value
        return grad_X

class TransformerModel(object):

    def __init__(self, Vocabulary, ContextSize, EmbeddingSize, HeadSize):
        self.vocabulary = Vocabulary
        self.contextSize = ContextSize

        self.token_embedding = np.random.randn(Vocabulary, EmbeddingSize) / np.sqrt(EmbeddingSize)
        self.position_embedding = np.random.randn(ContextSize, EmbeddingSize) / np.sqrt(EmbeddingSize)
        self.token_embedding_grad = np.zeros_like(self.token_embedding)
        self.position_embedding_grad = np.zeros_like(self.position_embedding)

        self.attentionHead = AttentionHead(HeadSize, EmbeddingSize, ContextSize)
        self.languageHead = LinearHead(HeadSize, Vocabulary)

    def SaveModel(self, path, filename="model.npz"):
        os.makedirs(path, exist_ok=True)
        np.savez_compressed(
            os.path.join(path, filename), 
            TokenEmbedding=self.token_embedding,
            PositionEmbedding=self.position_embedding,
            A1=self.attentionHead,
            L1=self.languageHead,
        )

    def LoadModel(self, path, filename="model.npz"):
        Data = np.load(os.path.join(path, filename), allow_pickle=True)
        self.token_embedding = Data['TokenEmbedding']
        self.position_embedding = Data['PositionEmbedding']
        self.attentionHead = Data['A1'].item()
        self.languageHead = Data['L1'].item()

    def Forward(self, idx):
        T = len(idx)
        self.last_indices = idx.copy()
        
        tok_embed = self.token_embedding[idx]
        pos_embed = self.position_embedding[np.arange(T)]

        x = tok_embed + pos_embed
        x = self.attentionHead.Forward(x)
        logits = self.languageHead.Forward(x)

        if(np.isnan(np.sum(logits))): raise Exception("NaN Encountered")

        return logits
    
    def Backward(self, grad_logits):
        grad_att_out = self.languageHead.Backward(grad_logits)
        grad_x = self.attentionHead.Backward(grad_att_out)

        for i, idx in enumerate(self.last_indices):
            self.token_embedding_grad[idx] += grad_x[i]

        for i in range(len(grad_x)):
            self.position_embedding_grad[i] += grad_x[i]

    def UpdateParameters(self, learning_rate):
        # Apply the Gradients
        self.token_embedding -= learning_rate * self.token_embedding_grad
        self.position_embedding -= learning_rate * self.position_embedding_grad

        self.attentionHead.key.weights -= learning_rate * self.attentionHead.key.grad_weights
        self.attentionHead.query.weights -= learning_rate * self.attentionHead.query.grad_weights
        self.attentionHead.value.weights -= learning_rate * self.attentionHead.value.grad_weights

        self.languageHead.weights -= learning_rate * self.languageHead.grad_weights
        self.languageHead.biases -= learning_rate * self.languageHead.grad_bias

        # Reset gradients after the update.
        self.token_embedding_grad.fill(0)
        self.position_embedding_grad.fill(0)
        self.attentionHead.key.grad_weights.fill(0)
        self.attentionHead.query.grad_weights.fill(0)
        self.attentionHead.value.grad_weights.fill(0)
        self.languageHead.grad_weights.fill(0)

    def Train(self, Input, Output, LearningRate=0.01):
        logits = self.Forward(Input)
        probs = Softmax(logits, axis=1)

        loss = CrossEntropy(probs, Output) / Input.shape[0]

        grad_logits = CrossEntropyDerivative(probs, Output) / Input.shape[0]

        self.Backward(grad_logits)
        self.UpdateParameters(LearningRate)

        return loss

    def Generate(self, Start, MaxTokens):
        for _ in range(MaxTokens):
            context = Start[-self.contextSize:]
            logits = self.Forward(context)[-1:]
            probs = Softmax(logits.T, axis=0).T
            pred = np.random.choice(self.vocabulary, p=probs.ravel())
            Start.append(pred)
            if(pred == 0): break
        return Start
            
