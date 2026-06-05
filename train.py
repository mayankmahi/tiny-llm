import self
import torch
import torch.nn as nn
import torch.nn.functional as F

# Training text
text = "hello world"

#Vocabulary
chars = sorted(list(text))

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

vocab_size = len(chars)

# Encoder
def encode(s):
    return [stoi[c] for c in s]

# Convert to tensor
data = torch.tensor(encode(text), dtype=torch.long)

#create traninng data
x = data[:-1]
y = data[1:]

#Tiny LLM
class TinyLLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.token_embeddings = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        #Prediction
        logits = self.token_embeddings(idx)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)

            return logits, loss

#Create Model
model = TinyLLM(vocab_size)

#optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

#Tranning Loop
for step in range(100):

    #Forward pass
    logits, loss = model(x,y)

    #Clear Old Gradients
    optimizer.zero_grad(set_to_none=True)

    #Backpropagation
    loss.backward()
    optimizer.step()
    if step % 10 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f}")

        # Final loss
        print("\nTraining Complete")
        print("Final Loss:", loss.item())




