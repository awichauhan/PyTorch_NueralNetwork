import torch

X = torch.tensor([
    [1,2,3,4],
    [1,2,3,4]
])

Y = torch.tensor([
    [4,5,6,7],
    [4,5,6,7]
])

Z = Y - X
print(Z)

A = X+Y
print(A)

B = X*Y   # element wise multiplication (not matrix multiplication)
print(B)

# Transpose and matrix multiplication (dot product)
M = torch.matmul(X,Y.T)  # inner dimension is not matching if we do (X,Y)
print(M)

print("Shape of X: ", X.shape)
R = X.reshape(4,2)
print("Shape of R: ", R.shape)