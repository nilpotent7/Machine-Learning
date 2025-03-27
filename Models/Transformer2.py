import os
from tqdm import tqdm
import numpy as np

def Softmax(X, axis=1):
    X_safe = np.clip(X - np.max(X, axis=axis, keepdims=True), -100, 100)  # Clip to avoid overflow
    e_x = np.exp(X_safe)
    return e_x / (np.sum(e_x, axis=axis, keepdims=True) + 1e-10)  # Add small epsilon to avoid division by zero

def CrossEntropy(probs, y):
    return -np.sum(np.log(probs[np.arange(len(y)), y]))

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
    def __init__(self, InSize, OutSize, Biases=True):
        self.useBiases = Biases
        self.weights = np.random.randn(OutSize, InSize) / np.sqrt(InSize)
        if(self.useBiases): self.biases = np.zeros(OutSize)
        else: self.biases = 0

    def Forward(self, X):
        self.Last_X = X.copy()
        return np.dot(X, self.weights.T) + self.biases

    def Backward(self, grad_output):
        original_grad_shape = grad_output.shape
        original_x_shape = self.Last_X.shape
        
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
        x_2d = self.Last_X.reshape(-1, self.Last_X.shape[-1])
        
        self.grad_weights = np.dot(grad_output_2d.T, x_2d)
        
        if(self.useBiases):
            self.grad_biases = np.sum(grad_output_2d, axis=0)
        else:
            self.grad_biases = 0
        
        grad_input = np.dot(grad_output_2d, self.weights)
        
        grad_input = grad_input.reshape(original_x_shape)
        
        return grad_input

class AttentionHead(object):

    def __init__(self, HeadSize, EmbeddingSize, ContextSize):
        self.key = LinearHead(EmbeddingSize, HeadSize, Biases=False)
        self.query = LinearHead(EmbeddingSize, HeadSize, Biases=False)
        self.value = LinearHead(EmbeddingSize, HeadSize, Biases=False)
        self.tril = np.tril(np.ones((ContextSize, ContextSize)))
    
    def Forward(self, X):
        self.Last_X = X.copy()
        B, T, C = X.shape
        
        k = self.key.Forward(X)
        q = self.query.Forward(X)
        v = self.value.Forward(X)
        
        d_k = k.shape[-1]
        W = q @ np.transpose(k, (0, 2, 1)) / np.sqrt(d_k)
        
        mask = self.tril[:T, :T] == 0
        mask_expanded = np.broadcast_to(mask, (B, T, T))
        W = np.where(mask_expanded, -np.inf, W)

        self.attention_weights = Softmax(W, axis=-1)
        
        output = np.matmul(self.attention_weights, v)
        
        return output
    
    def Backward(self, grad_output):
        B, T, H = grad_output.shape
        
        self.value.Last_X = self.Last_X
        grad_v = np.matmul(self.attention_weights.transpose(0, 2, 1), grad_output)
        grad_value = self.value.Backward(grad_v)
        
        v = self.value.Forward(self.Last_X)
        
        grad_W = np.matmul(grad_output, v.transpose(0, 2, 1))
        
        A = self.attention_weights
        grad_softmax = A * (grad_W - np.sum(grad_W * A, axis=-1, keepdims=True))
        
        mask = self.tril[:T, :T] == 0
        mask_expanded = np.broadcast_to(mask, (B, T, T))
        grad_softmax = np.where(mask_expanded, 0, grad_softmax)
        
        d_k = v.shape[-1]
        grad_scaled = grad_softmax / np.sqrt(d_k)
        
        k = self.key.Forward(self.Last_X)
        q = self.query.Forward(self.Last_X)
        
        grad_q = np.matmul(grad_scaled, k)
        grad_k = np.matmul(grad_scaled.transpose(0, 2, 1), q)
        
        self.query.Last_X = self.Last_X
        self.key.Last_X = self.Last_X
        
        grad_input_q = self.query.Backward(grad_q)
        grad_input_k = self.key.Backward(grad_k)
        
        grad_input = grad_input_q + grad_input_k + grad_value
        
        return grad_input

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
        
    def SaveParameters(self, filepath):
        """Save model parameters to a file"""
        try:
            np.savez(
                filepath,
                token_embedding=self.token_embedding,
                position_embedding=self.position_embedding,
                key_weights=self.attentionHead.key.weights,
                query_weights=self.attentionHead.query.weights,
                value_weights=self.attentionHead.value.weights,
                language_weights=self.languageHead.weights,
                language_biases=self.languageHead.biases if self.languageHead.useBiases else np.array([]),
                vocab_size=np.array([self.vocabulary]),
                context_size=np.array([self.contextSize]),
                embedding_size=np.array([self.token_embedding.shape[1]]),
                head_size=np.array([self.attentionHead.key.weights.shape[0]])
            )
            return True
        except Exception as e:
            print(f"Error saving model parameters: {e}")
            return False
        
    def LoadParameters(self, filepath):
        """Load model parameters from a file"""
        try:
            data = np.load(filepath)
            
            # Check if model architecture matches
            if data['vocab_size'][0] != self.vocabulary:
                raise ValueError(f"Vocabulary size mismatch: {data['vocab_size'][0]} vs {self.vocabulary}")
            if data['context_size'][0] != self.contextSize:
                raise ValueError(f"Context size mismatch: {data['context_size'][0]} vs {self.contextSize}")
            if data['embedding_size'][0] != self.token_embedding.shape[1]:
                raise ValueError(f"Embedding size mismatch: {data['embedding_size'][0]} vs {self.token_embedding.shape[1]}")
            if data['head_size'][0] != self.attentionHead.key.weights.shape[0]:
                raise ValueError(f"Head size mismatch: {data['head_size'][0]} vs {self.attentionHead.key.weights.shape[0]}")
            
            # Load parameters
            self.token_embedding = data['token_embedding']
            self.position_embedding = data['position_embedding']
            self.attentionHead.key.weights = data['key_weights']
            self.attentionHead.query.weights = data['query_weights']
            self.attentionHead.value.weights = data['value_weights']
            self.languageHead.weights = data['language_weights']
            
            if self.languageHead.useBiases and data['language_biases'].size > 0:
                self.languageHead.biases = data['language_biases']
            
            return True
        except Exception as e:
            print(f"Error loading model parameters: {e}")
            return False
        
    @classmethod
    def FromSavedModel(cls, filepath):
        """Create a new model instance from saved parameters"""
        try:
            data = np.load(filepath)
            vocab_size = int(data['vocab_size'][0])
            context_size = int(data['context_size'][0])
            embedding_size = int(data['embedding_size'][0])
            head_size = int(data['head_size'][0])
            
            model = cls(vocab_size, context_size, embedding_size, head_size)
            success = model.LoadParameters(filepath)
            
            if success:
                return model
            else:
                print("Failed to load parameters into model")
                return None
        except Exception as e:
            print(f"Error creating model from saved file: {e}")
            return None

    def Forward(self, idx):
        # Ensure we have batch dimension even with single sequences
        if isinstance(idx, list):
            idx = np.array(idx)
        
        # Add batch dimension if needed
        if idx.ndim == 1:
            idx = np.expand_dims(idx, axis=0)
            
        B, T = idx.shape
        self.last_indices = idx.copy()
        
        # Ensure token indices are properly broadcasted for embedding lookup
        tok_embed = np.zeros((B, T, self.token_embedding.shape[1]))
        for b in range(B):
            tok_embed[b] = self.token_embedding[idx[b]]
            
        # Position embeddings are the same for all sequences in batch
        pos_embed = self.position_embedding[np.arange(T)]
        pos_embed = np.expand_dims(pos_embed, axis=0)
        pos_embed = np.broadcast_to(pos_embed, (B, T, pos_embed.shape[2]))

        x = tok_embed + pos_embed
        x = self.attentionHead.Forward(x)
        logits = self.languageHead.Forward(x)

        # Check for NaN and handle it
        if np.isnan(np.sum(logits)):
            print("NaN detected in logits. Applying safe handling...")
            # Replace NaN values with zeros
            logits = np.nan_to_num(logits, nan=0.0)
            
        return logits
    
    def Backward(self, grad_output):
        B, T, V = grad_output.shape
        
        grad_x = self.languageHead.Backward(grad_output)
        grad_x = self.attentionHead.Backward(grad_x)
        
        token_indices = self.last_indices
        
        if grad_x.ndim == 2:
            np.add.at(self.token_embedding_grad, token_indices, grad_x)
            
            for pos in range(len(token_indices)):
                self.position_embedding_grad[pos] += grad_x[pos]
        else:
            for b in range(B):
                np.add.at(self.token_embedding_grad, token_indices[b], grad_x[b])
                
                for pos in range(len(token_indices[b])):
                    self.position_embedding_grad[pos] += grad_x[b, pos]
        
        return grad_x
    

    def Generate(self, Start, MaxTokens):
        # Convert to numpy array if not already
        if isinstance(Start, list):
            Start = np.array(Start)
            
        generated = Start.tolist()
        
        for _ in range(MaxTokens):
            # Get context with proper shape
            context = np.array(generated[-self.contextSize:])
            if len(context) < self.contextSize:
                # Pad with zeros if needed
                padding = np.zeros(self.contextSize - len(context), dtype=np.int32)
                context = np.concatenate([padding, context])
                
            # Forward pass
            logits = self.Forward(context)
            
            # Get the last token prediction
            next_token_logits = logits[0, -1, :]
            
            # Apply temperature
            temperature = 0.8
            next_token_logits = next_token_logits / temperature
            
            # Convert to probabilities
            probs = Softmax(np.expand_dims(next_token_logits, 0), axis=1)[0]
            
            # Sample from distribution
            try:
                pred = np.random.choice(self.vocabulary, p=probs)
            except ValueError:
                # If probabilities don't sum to 1 due to numerical issues
                probs = probs / np.sum(probs)  # Renormalize
                pred = np.random.choice(self.vocabulary, p=probs)
                
            # Add prediction to generated sequence
            generated.append(int(pred))
            
            # Stop if we generated the end token
            if pred == 0:
                break
                
        return generated
        
    def UpdateParameters(self, learning_rate, clip_value=1.0):
        # Clip gradients to prevent exploding gradients
        np.clip(self.token_embedding_grad, -clip_value, clip_value, out=self.token_embedding_grad)
        np.clip(self.position_embedding_grad, -clip_value, clip_value, out=self.position_embedding_grad)
        
        self.token_embedding -= learning_rate * self.token_embedding_grad
        self.token_embedding_grad.fill(0)
        
        self.position_embedding -= learning_rate * self.position_embedding_grad
        self.position_embedding_grad.fill(0)
        
        # Clip gradients for attention weights
        np.clip(self.attentionHead.key.grad_weights, -clip_value, clip_value, 
                out=self.attentionHead.key.grad_weights)
        np.clip(self.attentionHead.query.grad_weights, -clip_value, clip_value, 
                out=self.attentionHead.query.grad_weights)
        np.clip(self.attentionHead.value.grad_weights, -clip_value, clip_value, 
                out=self.attentionHead.value.grad_weights)
        
        self.attentionHead.key.weights -= learning_rate * self.attentionHead.key.grad_weights
        self.attentionHead.query.weights -= learning_rate * self.attentionHead.query.grad_weights
        self.attentionHead.value.weights -= learning_rate * self.attentionHead.value.grad_weights
        
        # Clip gradients for language head
        np.clip(self.languageHead.grad_weights, -clip_value, clip_value, 
                out=self.languageHead.grad_weights)
        
        self.languageHead.weights -= learning_rate * self.languageHead.grad_weights
        if self.languageHead.useBiases:
            if hasattr(self.languageHead, 'grad_biases'):
                np.clip(self.languageHead.grad_biases, -clip_value, clip_value, 
                       out=self.languageHead.grad_biases)
                self.languageHead.biases -= learning_rate * self.languageHead.grad_biases

# Example training function
def TrainTransformerExample():
    # Load a model
    loaded_model = TransformerModel.FromSavedModel("model.npz")

    # Sample text for training
    sample_text = "In natural language processing, a word embedding is a representation of a word. The embedding is used in text analysis. Typically, the representation is a real-valued vector that encodes the meaning of the word in such a way that the words that are closer in the vector space are expected to be similar in meaning."
    
    # Initialize tokenizer
    tokenizer = WordBasedTokenizer(sample_text)
    vocab_size = tokenizer.GetLength()
    print(f"Vocabulary size: {vocab_size}")
    
    # Model parameters - use even smaller dimensions
    context_size = 16
    embedding_size = 16   # Smaller embedding dimension
    head_size = 128        # Smaller head size
    
    # Initialize the model
    # model = TransformerModel(vocab_size, context_size, embedding_size, head_size)
    model = loaded_model
    
    # Tokenize the text
    tokens = tokenizer.Tokenize(sample_text)
    print(f"Number of tokens: {len(tokens)}")
    
    # Training parameters
    learning_rate = 0.03  # Very small learning rate
    epochs = 1000
    clip_value = 1.0      # Very aggressive gradient clipping
    batch_size = 10        # Batch size for training
    
    # Save file paths
    model_save_path = "model.npz"

    all_sequences = []
    all_targets = []
    total_loss = 0
    num_batches = 0

    for i in range(0, len(tokens) - context_size):
        x = tokens[i:i+context_size]
        y = tokens[i+1:i+context_size+1]
        all_sequences.append(x)
        all_targets.append(y)

    for batch_start in range(0, len(all_sequences), batch_size):
        batch_end = min(batch_start + batch_size, len(all_sequences))
        batch_size_actual = batch_end - batch_start
        
        x_batch = np.array(all_sequences[batch_start:batch_end])
        y_batch = np.array(all_targets[batch_start:batch_end])
        
        logits = model.Forward(x_batch)
        
        probs = Softmax(logits, axis=2)
        
        loss = 0
        for b in range(batch_size_actual):
            for t in range(context_size):
                position_probs = probs[b, t]
                target_index = y_batch[b, t]
                position_probs = np.maximum(position_probs, 1e-10)
                
                position_probs = position_probs.reshape(1, -1)
                target_indices = np.array([target_index])
                position_loss = CrossEntropy(position_probs, target_indices)
                loss += position_loss
        
        total_loss += loss
        num_batches += 1

    # Training loop
    best_loss = total_loss / num_batches
    best_epoch = 0
    
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        total_loss = 0
        num_batches = 0
        
        # Prepare all possible sequences for batching
        all_sequences = []
        all_targets = []
        
        for i in range(0, len(tokens) - context_size):
            # Input: sequence of tokens
            x = tokens[i:i+context_size]
            # Target: next token after the sequence
            y = tokens[i+1:i+context_size+1]
            
            all_sequences.append(x)
            all_targets.append(y)
            
        # Process in batches
        for batch_start in range(0, len(all_sequences), batch_size):
            batch_end = min(batch_start + batch_size, len(all_sequences))
            batch_size_actual = batch_end - batch_start
            
            # Create current batch
            x_batch = np.array(all_sequences[batch_start:batch_end])
            y_batch = np.array(all_targets[batch_start:batch_end])
            
            # Forward pass
            logits = model.Forward(x_batch)
            
            # Calculate probabilities with safe softmax
            probs = Softmax(logits, axis=2)
            
            # Calculate loss using the CrossEntropy function
            loss = 0
            for b in range(batch_size_actual):
                for t in range(context_size):
                    # Extract probabilities for this position
                    position_probs = probs[b, t]
                    target_index = y_batch[b, t]
                    
                    # Ensure no zeros in probabilities
                    position_probs = np.maximum(position_probs, 1e-10)
                    
                    # Use the predefined CrossEntropy function
                    # We need to reshape to match the expected format
                    position_probs = position_probs.reshape(1, -1)
                    target_indices = np.array([target_index])
                    position_loss = CrossEntropy(position_probs, target_indices)
                    loss += position_loss
            
            # Average loss across batch
            loss = loss / (batch_size_actual * context_size)
            
            # Check for NaN in loss
            if np.isnan(loss):
                print(f"NaN loss detected at epoch {epoch+1}, skipping batch")
                continue
                
            # Prepare gradient for backward pass (derivative of CrossEntropy with respect to logits)
            grad_output = np.zeros_like(probs)
            for b in range(batch_size_actual):
                for t in range(context_size):
                    # The gradient of CrossEntropy is (probs - one_hot_target)
                    # Copy the probabilities
                    grad_output[b, t] = probs[b, t].copy()
                    # Subtract 1 from the target index (equivalent to subtracting one-hot vector)
                    grad_output[b, t, y_batch[b, t]] -= 1.0
            
            # Scale gradients by batch size for consistent updates
            grad_output /= (batch_size_actual * context_size)
            
            # Backward pass
            model.Backward(grad_output)
            
            # Update parameters with clipping
            model.UpdateParameters(learning_rate, clip_value=clip_value)
            
            total_loss += loss
            num_batches += 1
        
        # Print progress
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            
            # Save best model info
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch + 1
                
                model.SaveParameters(model_save_path)
                
            if (epoch + 1) % 10 == 0 or epoch == 0:
                os.system('cls' if os.name == 'nt' else 'clear')
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.8f}" + 
                     (f" (best: {best_loss:.8f} at epoch {best_epoch})" if epoch > 0 else ""))
    
    print(f"Training complete! Best loss: {best_loss:.4f} at epoch {best_epoch}")
    print(f"Best model saved to {model_save_path}")

    prompt = "In natural language"
    
    start_tokens = tokenizer.Tokenize(prompt)[:-1]
    print(f"\nGenerating text from prompt: '{prompt}'")
    generated = model.Generate(start_tokens, MaxTokens=15)
    
    # Convert tokens to words
    generated_words = tokenizer.Untokenize(generated)

    print(f"Generated text: {''.join(generated_words).strip()}")

# Run the example if this file is executed directly
if __name__ == "__main__":
    TrainTransformerExample()
