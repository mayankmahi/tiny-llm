import torch
import torch.nn as nn
import torch.nn.functional as F

sampleText = "Hello World"

# Vocabulary
chars = sorted(list(set(sampleText)))

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

vocab_size = len(chars)

#Encoder
def encode(s):
    return [stoi[ch] for ch in s]

#convert to tensor
data = torch.tensor(encode(sampleText), dtype=torch.long)

#Tiny Model
class TinyLLM(nn.Module):
    def __init__(self, vocab_siz):
        super().__init__()

        #Embedding Table
        self.token_embeddings = nn.Embedding(vocab_siz, vocab_siz)
    def forward(self, idx):

        #Get Embedding
        logits = self.token_embeddings(idx)

        return logits

#Create Model
model = TinyLLM(vocab_size)

x = data[:4]

print("Input Tokens:")
print(x)

print("\nInput Shape:")
print(x.shape)

logits = model(x)

print("\nLogits:")
print(logits)

print("\nLogits Shape:")
print(logits.shape)

