import os
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork(object):

    # Initialize random Weights & Biases
    def __init__(self, sizes, HiddenActivation, FinalActivation, LossFunction):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases  = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.ActivateHidden = HiddenActivation
        self.ActivateFinal = FinalActivation
        self.Loss = LossFunction

        if(FinalActivation == self.Softmax and LossFunction != self.CrossEntropyLoss):
            raise Exception("Softmax requires CrossEntropyLoss for efficient derivative calculation")

    def UseSGD(self, LearningRate, DecayFunction = None, DecayRate = None):
        self.optimizer = self.SGD
        self.learning_rate = LearningRate
        self.decay = DecayFunction if DecayFunction else self.IdentityDecay
        self.rate = DecayRate

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

    # Sigmoid Activation Function & its Derivative
    @staticmethod
    def Sigmoid(z, Derivative):
        if(Derivative): return z * (1 - z)
        else: return 1.0 / (1.0 + np.exp(-z))

    # Softmax Activation Function. Derivative is avoided when its coupled with CrossEntropyLoss
    @staticmethod
    def Softmax(z, Derivative):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    # Mean Squared Error Loss Function & its Derivative
    @staticmethod
    def MeanSquaredLoss(result, desired, Derivative):
        if(Derivative): return 2 * (np.array(result) - np.array(desired))
        else: return (np.array(result) - np.array(desired))**2

    # Cross Entropy Loss Function & its Derivative
    @staticmethod
    def CrossEntropyLoss(result, desired, Derivative):
        epsilon = 1e-12
        if(Derivative): return result - desired
        else: return -np.sum(desired * np.log(result + epsilon))

    # Forward Pass, storing output of each layer
    def FeedForward(self, a):
        activations = [a]
        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            z = np.dot(w, a) + b
            a = self.ActivateFinal(z, False) if (i == len(self.weights) - 1) else self.ActivateHidden(z, False)
            activations.append(a)
        return activations

    # Make a prediction on given input
    def Evaluate(self, a):
        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            z = np.dot(w, a) + b
            a = self.ActivateFinal(z, False) if (i == len(self.weights) - 1) else self.ActivateHidden(z, False)
        return a

    # Perform Backpropogation using the selected optimizer algorithm
    def Train(self, Input, Desired, Epoch):
        self.optimizer(Input, Desired, Epoch)
    
    # Stochastic Gradient Descent Algorithm
    def SGD(self, Input, Desired, Epoch):
        activations = self.FeedForward(Input)

        # Gradient = (Derivative of Loss) * (Derivative of Activation)
        delta = self.Loss(activations[-1], Desired, True) * self.ActivateFinal(activations[-1], True)
        deltas = [delta]
        
        # Apply gradient to each layer's output to get delta
        for l in range(2, self.num_layers):
            sp = self.ActivateHidden(activations[-l], True)
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            deltas.insert(0, delta)
        
        # Apply delta to each layer's parameters to adjust preferrable
        for i in range(len(self.weights)):
            self.weights[i] -= self.decay(self.learning_rate, self.rate, Epoch) * np.dot(deltas[i], activations[i].T)
            self.biases[i]  -= self.decay(self.learning_rate, self.rate, Epoch) * deltas[i]

    # AdamW Algorithm
    def AdamW(self, Input, Desired, Epoch):
        activations = self.FeedForward(Input)
        
        # Gradient = (Derivative of Loss) * (Derivative of Activation)
        delta = self.Loss(activations[-1], Desired, True) * self.ActivateFinal(activations[-1], True)
        deltas = [delta]
        
        # Apply gradient to each layer's output to get delta
        for l in range(2, self.num_layers):
            sp = self.ActivateHidden(activations[-l], True)
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
            self.weights[i] -= self.decay(self.learning_rate, self.rate, Epoch) * update_w
            self.biases[i]  -= self.decay(self.learning_rate, self.rate, Epoch) * update_b

    def FindFiles(self, Path):
        file_paths = []
        for root, _, files in os.walk(Path):
            for file in files:
                file_paths.append(os.path.join(root, file))
        return file_paths

    # Save Weights & Biases as Numpy files
    def SaveData(self, path):
        os.makedirs(path, exist_ok=True)
        for k, x in enumerate(self.weights):
            np.save(os.path.join(path, f"weights{k}.npy"), x)
        for k, x in enumerate(self.biases):
            np.save(os.path.join(path, f"biases{k}.npy"), x)
    
    # Save Weights & Biases as an Image
    def SaveDataAsImage(self, path):
        os.makedirs(path, exist_ok=True)
        
        for k, x in enumerate(self.weights):
            plt.imshow(x, cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.title(f'Weights {k}')
            plt.savefig(os.path.join(path, f"weights{k}.png"))
            plt.close()
        
        for k, x in enumerate(self.biases):
            plt.figure(figsize=(len(x), 1))
            plt.imshow(x.reshape(1, -1), cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.title(f'Biases {k}')
            plt.savefig(os.path.join(path, f"biases{k}.png"))
            plt.close()

    # Load Weights & Biases from Numpy files
    def LoadData(self, path):
        allFiles = self.FindFiles(path)
        weightsFiles = [x for x in allFiles if "weights" in x]
        biasesFiles = [x for x in allFiles if "biases" in x]
        self.weights = [np.load(w) for w in weightsFiles]
        self.biases = [np.load(b) for b in biasesFiles]