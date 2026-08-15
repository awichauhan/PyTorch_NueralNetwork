import torch
import torch.nn.functional as F

# --------------------------------
# 1. Training data
# --------------------------------

X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

Y = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [1.0]
])


# --------------------------------
# 2. Initialize hidden layer
# --------------------------------

# 2 input features -> 3 hidden neurons

W1 = torch.tensor([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6]
], requires_grad=True)

B1 = torch.tensor(
    [0.0, 0.0, 0.0],
    requires_grad=True
)

# 3 hidden neurons -> 1 output neuron

W2 = torch.tensor([
    [0.2],
    [0.3],
    [0.4]
], requires_grad=True)

B2 = torch.tensor(
    [0.0],
    requires_grad=True
)


learning_rate = 0.1
epochs = 1000


# --------------------------------
# 4. Training loop
# --------------------------------

for epoch in range(epochs):

    # Hidden layer
    Z1 = X @ W1 + B1
    A1 = torch.sigmoid(Z1)

    # Output layer
    Z2 = A1 @ W2 + B2
    A2 = torch.sigmoid(Z2)

    # Loss
    loss = F.binary_cross_entropy(A2, Y)

    # Backpropagation
    loss.backward()

    # Update parameters
    with torch.no_grad():

        W1 -= learning_rate * W1.grad
        B1 -= learning_rate * B1.grad

        W2 -= learning_rate * W2.grad
        B2 -= learning_rate * B2.grad

    # Clear gradients
    W1.grad.zero_()
    B1.grad.zero_()

    W2.grad.zero_()
    B2.grad.zero_()

    if epoch % 100 == 0:
        print(
            "Epoch:", epoch,
            "| Loss:", loss.item()
        )


# --------------------------------
# 5. Final prediction
# --------------------------------

Z1 = X @ W1 + B1
A1 = torch.sigmoid(Z1)

Z2 = A1 @ W2 + B2
A2 = torch.sigmoid(Z2)

predictions = (A2 >= 0.5).float()


print("\nFinal probabilities:")
print(A2)

print("\nPredictions:")
print(predictions)

print("\nActual:")
print(Y)