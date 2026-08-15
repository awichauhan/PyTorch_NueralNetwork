import torch
import torch.nn as nn


X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])


Y = torch.tensor([   # XOR table
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])


class DeepNeuralNetwork(nn.Module):  # base class used to define an entire neural network structure
    def __init__(self):   # need to understand from python concept level
        super().__init__()

        self.layer1 = nn.Linear(2, 4)      # defines W & B and performs forward pass (Z = W.X + B)
        self.layer2 = nn.Linear(4, 3)  # 4 input features and 3 output neurons is basically W.shape = (3,4) and B.shape = (3,)
        self.layer3 = nn.Linear(3,3)
        self.layer4 = nn.Linear(3,3)
        self.output_layer = nn.Linear(3, 1)

    def forward(self, X):

        Z1 = self.layer1(X)   # defined W and B and performed forward calculation
        A1 = torch.relu(Z1)

        Z2 = self.layer2(A1)
        A2 = torch.relu(Z2)

        Z3 = self.layer3(A2)
        A3 = torch.relu(Z3)

        Z4 = self.layer4(A3)
        A4 = torch.relu(Z4)

        Z5 = self.output_layer(A4)

        """
        # A4 using reLU will give logits like -3.2, 4,1 (not probabilities like sigmoid) so we need to pass this raw Z5 logit directly for Binary Cross Entropy loss 
        using logits (BCEwithLogitsLoss). It internally applies sigmoid and calculates BCE loss.
        """

        return Z5

def main():
    model = DeepNeuralNetwork()    # created object of deep-neural network clacc
    criterion = nn.BCEWithLogitsLoss()  # created object of BCELoss class
    optimizer = torch.optim.SGD (model.parameters(), lr=0.1)   # created object of in-built gradient descent optimizer class

    """
    model.parameters() = all trainable weights and biases in the model
    lr=0.1 = learning rate
    SGD = Stochastic Gradient Descent
    """

    epochs = 5000

    for epoch in range(epochs):

        prediction = model(X)   #forward propagation (storing A3 inside prediction)
        loss = criterion(prediction, Y)   # passed prediction and Y to criterion object
        optimizer.zero_grad()   # function for clearing gradients parameters (weights and bias)
        loss.backward()   # gradient calculation
        optimizer.step()   # update parameters ; uses those gradients to update parameters.

        if epoch % 500 == 0:
            print(
                "Epoch:", epoch,
                "| Loss:", loss.item()
            )

    with torch.no_grad():
        logits = model(X)  # saving raw logits from ReLU
        probabilities = torch.sigmoid(logits)  # applying sigmoid and storing it in prediction
        predictions = (probabilities >= 0.5).float()   # then making prediction

    print("\nFinal probabilities:")
    print(probabilities)

    print("\nPredicted classes:")
    print(predictions)

    print("\nActual classes:")
    print(Y)

    print("\nLayer 1 Weights:")
    print(model.layer1.weight)

    print("\nLayer 1 Bias:")
    print(model.layer1.bias)

    print("\nLayer 2 Weights:")
    print(model.layer2.weight)

    print("\nLayer 2 Bias:")
    print(model.layer2.bias)

    print("\nLayer 3 Weights:")
    print(model.layer3.weight)

    print("\nLayer 3 Bias:")
    print(model.layer3.bias)

    print("\nLayer 4 Weights:")
    print(model.layer4.weight)

    print("\nLayer 4 Bias:")
    print(model.layer4.bias)

    print("\nOutput Layer Weights:")
    print(model.output_layer.weight)

    print("\nOutput Layer Bias:")
    print(model.output_layer.bias)

main()
