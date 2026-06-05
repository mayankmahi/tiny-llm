#Sample Training text
text = """
hello world
tiny llm learning
"""

# Step 1: Create vocabulary
chars = sorted(list(set(text)))

print("chars" , chars)
print("vocabulary", len(chars))

#step 2: Create mapping
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

print('characters :', len(chars))
print('vocabulary :', len(chars))
print(stoi)
print(itos)

#encoder function
def encode(s):
    return [stoi[c] for c in s]

#decoder function
def decode(l):
    return [itos[i] for i in l]

encoded = "hello world"
print("encode : ", encoded)

decoded = encode(encoded)
print("decode : ", decoded)

