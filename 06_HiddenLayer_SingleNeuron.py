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

W1 = torch.tensor([0.5], requires_grad=True)
B1 = torch.tensor([0.0], requires_grad=True)
W2 = torch.tensor([0.8], requires_grad=True)
B2 = torch.tensor([0.0], requires_grad=True)
learning_rate = 0.1
epochs = 1000


# --------------------------------
# 3. Training loop
# --------------------------------

for epoch in range(epochs):

    # Forward propagation
    Z1 = X * W1 + B1

    A1 = torch.sigmoid(Z1)

    Z2 =A1 * W2 + B2
    A2 = torch.sigmoid(Z2)


    # Loss calculation
    loss = F.binary_cross_entropy(A2, Y)


    # Backpropagation
    loss.backward()


    # --------------------------------
    # 4. Update parameters
    # --------------------------------

    with torch.no_grad():  # don't track the parameter update as another computation graph in W or B tensors

        W2 -= learning_rate * W2.grad
        B2 -= learning_rate * B2.grad

        W1 -= learning_rate * W1.grad
        B1-= learning_rate * B1.grad


    # --------------------------------
    # 5. Clear old gradients
    # --------------------------------

    W1.grad.zero_()  # clear old gradients so pytorch doesnt accumulate gradient value before next iteration
    B1.grad.zero_()
    W2.grad.zero_()
    B2.grad.zero_()

    # Print progress
    if epoch % 100 == 0:

        print(
            "Epoch:", epoch,
            "| Loss:", loss.item()
        )


# --------------------------------
# 6. Final prediction
# --------------------------------


Z1 = X * W1 + B1

A1 = torch.sigmoid(Z1)

Z2 = A1 * W2 + B2
A2 = torch.sigmoid(Z2)

predictions = (A2 >= 0.5).float()


print("\nHidden Layer W1:")
print(W1)

print("\nHidden Layer B1:")
print(B1)

print("\nHidden Layer Probability:")
print(A1)


print("\nOutputLayer W:")
print(W2)

print("\nOutputLayer B:")
print(B2)

print("\nOutputLayer Probability:")
print(A2)

print("\nPredictions:")
print(predictions)

print("\nActual values:")
print(Y)