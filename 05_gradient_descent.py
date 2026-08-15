import torch
import torch.nn.functional as F


# --------------------------------
# 1. Training data
# --------------------------------

X = torch.tensor([
    [0.0],
    [1.0],
    [2.0],
    [3.0]
])

Y = torch.tensor([
    [0.0],
    [0.0],
    [1.0],
    [1.0]
])


# --------------------------------
# 2. Initialize parameters
# --------------------------------

W = torch.tensor([0.5], requires_grad=True)
B = torch.tensor([0.0], requires_grad=True)

learning_rate = 0.1
epochs = 1000


# --------------------------------
# 3. Training loop
# --------------------------------

for epoch in range(epochs):

    # Forward propagation
    Z = X * W + B

    A = torch.sigmoid(Z)

    # Loss calculation
    loss = F.binary_cross_entropy(A, Y)


    # Backpropagation
    loss.backward()


    # --------------------------------
    # 4. Update parameters
    # --------------------------------

    with torch.no_grad():

        W -= learning_rate * W.grad
        B -= learning_rate * B.grad


    # --------------------------------
    # 5. Clear old gradients
    # --------------------------------

    W.grad.zero_()
    B.grad.zero_()


    # Print progress
    if epoch % 100 == 0:

        print(
            "Epoch:", epoch,
            "| Loss:", loss.item()
        )


# --------------------------------
# 6. Final prediction
# --------------------------------

Z = X * W + B
A = torch.sigmoid(Z)

predictions = (A >= 0.5).float()


print("\nFinal W:")
print(W)

print("\nFinal B:")
print(B)

print("\nProbabilities:")
print(A)

print("\nPredictions:")
print(predictions)

print("\nActual values:")
print(Y)