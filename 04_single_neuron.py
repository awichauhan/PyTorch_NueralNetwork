import torch
import torch.nn.functional as F

X = torch.tensor(2.0)
Y = torch.tensor(1.0)

W = torch.tensor(0.5, requires_grad=True)
B = torch.tensor(1.0, requires_grad=True)

Z = W * X + B
A = torch.sigmoid(Z)
loss = F.binary_cross_entropy(A, Y)
loss.backward()

print("dL/dw:", W.grad)
print("dL/db:", B.grad)


