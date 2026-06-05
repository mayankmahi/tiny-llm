import torch

SampleText = "Hello World"

chars = sorted(list(set(SampleText)))

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

#Encoder
def encode(s):
    return [stoi[c] for c in s]

#Convert text to number
data = torch.tensor(encode(SampleText), dtype=torch.long)
print("Encoded Data : ", data)

print("\ndataShape:")
print(data.shape)

#context window size
block_size = 4

#Create Input output pair
x = data[:block_size]
y = data[1:block_size+1]

print("\nTraining EXample : ")
for t in range(block_size):
    context = x[t]
    target = y[t]

    print(f"Input: {context.tolist()} --> Target: {target}")
