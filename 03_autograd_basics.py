import torch

x = torch.tensor(2.0, requires_grad=True)  #Track the operations involving x, and if .backward() is eventually called,
# calculate and store the gradient with respect to x

y = x**2

y.backward()  # it is going backward to entire computation graph and calculating derivatives and gradient of all variables

print("Example 1")
print("x:", x)
print("y:", y)
print("dy/dx:", x.grad)   #because x's gradient is already calculated by backward function

