from __future__ import print_function
import torch

# Create a randomly initialized matrix
x = torch.rand(5, 3)
print(x)

# Matrix filled with 0 and type long
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# Tensor directly from data
x = torch.tensor([5.5, 3])
print(x)