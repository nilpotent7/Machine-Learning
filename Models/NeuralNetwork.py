import os
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork(object):

    # Initialize random Weights & Biases
    def __init__(self, sizes, HiddenActivation, FinalActivation, LossFunction, DecayFunction = None, DecayRate = None):
        if(FinalActivation == self.Softmax and LossFunction != self.CrossEntropyLoss):
            raise Exception("Softmax requires CrossEntropyLoss for efficient derivative calculation")
        
        self.num_layers = len(sizes)
        self.sizes = sizes
        
        # Randomly initializing weights and biases
        self.biases  = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        self.ActivateHidden = HiddenActivation
        self.ActivateFinal = FinalActivation
        self.Loss = LossFunction
        self.decay = DecayFunction if DecayFunction else self.IdentityDecay
        self.decay_rate = DecayRate

        self.RescaleParameters()
        
    def RescaleParameters(self):
        for i in range(len(self.weights)):
            if(i == len(self.weights)-1): 0.01
            else: self.weights[i] *= self.ActivateHidden(None, 2) / self.weights[i].shape[0]**0.5

        for i in range(len(self.biases)):
            if(i == len(self.biases)-1): 0
            else: self.biases[i] *= 0.01

    def UseSGD(self, LearningRate):
        self.optimizer = self.SGD
        self.learning_rate = LearningRate

    def UseAdamW(self, LearningRate, Beta1=0.9, Beta2=0.999, Epsilon=1e-8, WeightDecay=0.01):
        self.optimizer = self.AdamW
        self.learning_rate = LearningRate
        self.beta1 = Beta1
        self.beta2 = Beta2
        self.epsilon = Epsilon
        self.weight_decay = WeightDecay

        self.time_step = 0
        self.m_weights = [np.zeros_like(w) for w in self.weights]
        self.v_weights = [np.zeros_like(w) for w in self.weights]
        self.m_biases  = [np.zeros_like(b) for b in self.biases]
        self.v_biases  = [np.zeros_like(b) for b in self.biases]

    # No Decay for Learning Rate (Identity Function) 
    @staticmethod
    def IdentityDecay(Initial, Rate, Epoch):
        return Initial

    # Exponential Learning Rate Decay Function
    @staticmethod
    def ExponentialDecay(Initial, Rate, Epoch):
        return Initial * np.exp(-Rate * Epoch)
    
    # Linear (Identity) Activation Function & its Derivative & its Gain (for standard deviation rescaling)
    @staticmethod
    def Linear(z, Type):
        if(Type == 1): return np.ones_like(z)
        if(Type == 2): return 1
        return z

    # Sigmoid Activation Function & its Derivative & its Gain (for standard deviation rescaling)
    @staticmethod
    def Sigmoid(z, Type):
        if(Type == 1): return z * (1 - z)
        if(Type == 2): return 1
        return 1.0 / (1.0 + np.exp(-z))

    # Tanh Activation Function & its Derivative & its Gain (for standard deviation rescaling)
    @staticmethod
    def Tanh(z, Type):
        if(Type == 1): return 1 - z**2
        if(Type == 2): return 5/3
        return np.tanh(z)

    # ReLU Activation Function & its Derivative & its Gain (for standard deviation rescaling)
    @staticmethod
    def ReLU(z, Type):
        if(Type == 1): return (z > 0).astype(float)
        if(Type == 2): return np.sqrt(2)
        return np.maximum(0, z)

    # Softmax Activation Function. Derivative is avoided when its coupled with CrossEntropyLoss
    @staticmethod
    def Softmax(z, Type):
        if(Type == 2): return 1
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    # Mean Squared Error Loss Function & its Derivative
    @staticmethod
    def MeanSquaredError(result, desired, Derivative):
        if(Derivative): return 2 * (np.array(result) - np.array(desired))
        else: return (np.array(result) - np.array(desired))**2

    # Cross Entropy Loss Function & its Derivative
    @staticmethod
    def CrossEntropyLoss(result, desired, Derivative):
        epsilon = 1e-12
        if(Derivative): return result - desired
        else: return -np.sum(desired * np.log(result + epsilon))

    def CurrentLearningRate(self, Epoch):
        return self.decay(self.learning_rate, self.decay_rate, Epoch)

    # Forward Pass, storing output of each layer
    def FeedForward(self, a):
        activations = [a]
        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            z = np.dot(w, a) + b
            if i == len(self.weights) - 1:
                a = self.ActivateFinal(z, False)
            else:
                a = self.ActivateHidden(z, False)
            activations.append(a)
        return activations

    # Make a prediction on given input
    def Evaluate(self, a):
        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            z = np.dot(w, a) + b
            if i == len(self.weights) - 1:
                a = self.ActivateFinal(z, False)
            else:
                a = self.ActivateHidden(z, False)
        return a

    # Perform Backpropogation using the selected optimizer algorithm
    def Train(self, Input, Desired, Epoch=1):
        return self.optimizer(Input, Desired, Epoch)
    
    # Stochastic Gradient Descent Algorithm with optional gradient output for the input.
    def SGD(self, Input, Desired, Epoch):
        activations = self.FeedForward(Input)
        
        # Compute gradient for output layer
        delta = self.Loss(activations[-1], Desired, True) * self.ActivateFinal(activations[-1], 1)
        deltas = [delta]
        
        # Backpropagate through hidden layers
        for l in range(2, self.num_layers):
            sp = self.ActivateHidden(activations[-l], 1)
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            deltas.insert(0, delta)
        
        # Update each layer's weights and biases
        for i in range(len(self.weights)):
            lr = self.decay(self.learning_rate, self.decay_rate, Epoch)
            self.weights[i] -= lr * np.dot(deltas[i], activations[i].T)
            self.biases[i]  -= lr * deltas[i]

        grad_input = np.dot(self.weights[0].T, deltas[0])
        return grad_input

    # AdamW Algorithm
    def AdamW(self, Input, Desired, Epoch):
        activations = self.FeedForward(Input)
        
        # Gradient = (Derivative of Loss) * (Derivative of Activation)
        delta = self.Loss(activations[-1], Desired, True) * self.ActivateFinal(activations[-1], 1)
        deltas = [delta]
        
        # Apply gradient to each layer's output to get delta
        for l in range(2, self.num_layers):
            sp = self.ActivateHidden(activations[-l], 1)
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            deltas.insert(0, delta)
        
        # Increment time step
        self.time_step += 1
        
        # Update parameters using AdamW update rule
        for i in range(len(self.weights)):
            # Compute gradients for current layer
            grad_w = np.dot(deltas[i], activations[i].T)
            grad_b = deltas[i]
            
            # Update biased first moment estimates
            self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * grad_w
            self.m_biases[i]  = self.beta1 * self.m_biases[i]  + (1 - self.beta1) * grad_b
            
            # Update biased second moment estimates
            self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * (grad_w ** 2)
            self.v_biases[i]  = self.beta2 * self.v_biases[i]  + (1 - self.beta2) * (grad_b ** 2)
            
            # Compute bias-corrected moment estimates
            m_hat_w = self.m_weights[i] / (1 - self.beta1 ** self.time_step)
            m_hat_b = self.m_biases[i]  / (1 - self.beta1 ** self.time_step)
            v_hat_w = self.v_weights[i] / (1 - self.beta2 ** self.time_step)
            v_hat_b = self.v_biases[i]  / (1 - self.beta2 ** self.time_step)
            
            # Compute AdamW update (decoupled weight decay)
            update_w = m_hat_w / (np.sqrt(v_hat_w) + self.epsilon) + self.weight_decay * self.weights[i]
            update_b = m_hat_b / (np.sqrt(v_hat_b) + self.epsilon) + self.weight_decay * self.biases[i]
            
            # Update parameters (subtract the update term)
            self.weights[i] -= self.decay(self.learning_rate, self.decay_rate, Epoch) * update_w
            self.biases[i]  -= self.decay(self.learning_rate, self.decay_rate, Epoch) * update_b

    def FindFiles(self, Path):
        file_paths = []
        for root, _, files in os.walk(Path):
            for file in files:
                file_paths.append(os.path.join(root, file))
        return file_paths

    # Save Weights & Biases as Numpy Zipped file
    def SaveData(self, path, filename="neurons.npz"):
        os.makedirs(path, exist_ok=True)
        w = np.array(self.weights, dtype=object)
        b = np.array(self.biases, dtype=object)
        np.savez_compressed(os.path.join(path, filename), weights=w, biases=b)
    
    # Save Weights & Biases as an Image
    def SaveDataAsImage(self, path, dpi=250):
        os.makedirs(path, exist_ok=True)
        
        for k, x in enumerate(self.weights):
            plt.imshow(x, cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.title(f'Weights {k}')
            plt.savefig(os.path.join(path, f"weights{k}.png"), dpi=dpi)
            plt.close()
        
        for k, x in enumerate(self.biases):
            plt.figure(figsize=(len(x), 1))
            plt.imshow(x.reshape(1, -1), cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.title(f'Biases {k}')
            plt.savefig(os.path.join(path, f"biases{k}.png"), dpi=dpi)
            plt.close()

    # Load Weights & Biases from Numpy Zipped File
    def LoadData(self, path, filename="neurons.npz"):
        data = np.load(os.path.join(path, filename), allow_pickle=True)
        self.weights = data['weights'].tolist()
        self.biases = data['biases'].tolist()