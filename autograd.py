import torch

# requires grad means that the computations on the tensor are tracked
x = torch.ones(2, 2, requires_grad=True)
print(x)

# executing a tensor operation
y = x + 2
print(y)

# show the function that has created the tensor
print(y.grad_fn)

z = y * y * 3
out = z.mean()
print(z, out)
