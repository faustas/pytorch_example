from __future__ import print_function
import torch

# Create a randomly initialized matrix
print("random tensor")
x = torch.rand(5, 3)
print(x)

# Matrix filled with 0 and type long
print("zeros tensor")
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# Tensor directly from data
print("tensor from data")
x = torch.tensor([5.5, 3])
print(x)

x = torch.ones(5, 3, dtype=torch.int)
print("ones")
print(x)
y = torch.zeros(5, 3, dtype=torch.int)
print("zeros")
print(y)
print("addition")
print(x + y)

# or as output tensor
print("with output tensor")
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

print("in place addition")
y.add_(x)
print(y)

print("resizing or reshaping tensors")
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())
print(y)

print("Get the item from an one element tensor")
x = torch.randn(1)
print(x)
print(x.item())

print("Converting a tensor to a numpy array")
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
print("The two structures share the same memory")
a.add_(1)
print(a)
print(b)

print("Numpy array to torch sensor")
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
