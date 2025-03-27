from Models.Transformer2 import TransformerModel, WordBasedTokenizer

sample_text = "In natural language processing, a word embedding is a representation of a word. The embedding is used in text analysis. Typically, the representation is a real-valued vector that encodes the meaning of the word in such a way that the words that are closer in the vector space are expected to be similar in meaning."
model = TransformerModel.FromSavedModel("model.npz")
tokenizer = WordBasedTokenizer(sample_text)
prompt = "natural language is used in a"
    
start_tokens = tokenizer.Tokenize(prompt)[:-1]
print(f"\nGenerating text from prompt: '{prompt}'")
generated = model.Generate(start_tokens, MaxTokens=50)

generated_words = tokenizer.Untokenize(generated)

print(f"Generated text: {''.join(generated_words).strip()}")