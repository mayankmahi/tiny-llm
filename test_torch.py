import torch
import numpy as np

#create Tensor
x = torch.tensor([1,2])
y = torch.tensor([3,4])

#tensor addition
z = x + y

print("tensor z = x + y", z)
print("tensor x", x)
print("tensor y", y)

#check GPU Avaibility
if torch.cuda.is_available():
    print("cuda is available")
    x = x.cuda()
    y = y.cuda()
    print("cuda version", torch.version.cuda)
    print("GPU Name", torch.cuda.get_device_name(0))
else:
    print("Running on CPU : ", torch.__version__)