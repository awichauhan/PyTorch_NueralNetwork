import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. DATA
# ============================================================

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
    [0.0]
])


"""
X            = all input examples; 4 Examples
X[index]     = one input example
XORDataset   = wraps all examples + labels
DataLoader   = groups individual examples into batches
"""
# ============================================================
# 2. CUSTOM DATASET
# ============================================================

class XORDataset(Dataset):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


# ============================================================
# 3. DEEP NEURAL NETWORK
# ============================================================

class DeepNeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(2, 4)
        self.layer2 = nn.Linear(4, 3)
        self.layer3 = nn.Linear(3, 3)
        self.layer4 = nn.Linear(3, 3)

        self.output_layer = nn.Linear(3, 1)

    def forward(self, X):

        Z1 = self.layer1(X)
        A1 = torch.relu(Z1)

        Z2 = self.layer2(A1)
        A2 = torch.relu(Z2)

        Z3 = self.layer3(A2)
        A3 = torch.relu(Z3)

        Z4 = self.layer4(A3)
        A4 = torch.relu(Z4)



        Z5 = self.output_layer(A4)

        # return logits
        return Z5


# ============================================================
# 4. XAVIER INITIALIZATION
# ============================================================

def initialize_xavier(model):

    for layer in model.modules():

        if isinstance(layer, nn.Linear):

            nn.init.xavier_uniform_(layer.weight)

            nn.init.zeros_(layer.bias)


# ============================================================
# 5. HE / KAIMING INITIALIZATION
# ============================================================

def initialize_he(model):

    for layer in model.modules():

        if isinstance(layer, nn.Linear):

            nn.init.kaiming_uniform_(
                layer.weight,
                nonlinearity="relu"
            )

            nn.init.zeros_(layer.bias)


# ============================================================
# 6. TRAINING FUNCTION
# ============================================================

def train_model(model, dataloader, model_name):

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1
    )

    epochs = 5000

    print("\nTraining:", model_name)

    for epoch in range(epochs):

        total_loss = 0

        # go batch by batch
        for X_batch, Y_batch in dataloader:

            logits = model(X_batch)

            loss = criterion(logits, Y_batch)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        if epoch % 500 == 0:

            print(
                "Epoch:",
                epoch,
                "| Loss:",
                total_loss
            )


# ============================================================
# 7. PREDICTION FUNCTION
# ============================================================

def evaluate_model(model, model_name):

    with torch.no_grad():

        logits = model(X)

        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= 0.5
        ).float()

    print("\nResults:", model_name)

    print("\nLogits:")
    print(logits)

    print("\nProbabilities:")
    print(probabilities)

    print("\nPredictions:")
    print(predictions)

    print("\nActual:")
    print(Y)


# ============================================================
# 8. MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = XORDataset(X, Y)

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True
    )

    # --------------------------------------------------------
    # DEFAULT MODEL
    # --------------------------------------------------------

    torch.manual_seed(42)

    model_default = DeepNeuralNetwork()

    # --------------------------------------------------------
    # XAVIER MODEL
    # --------------------------------------------------------

    torch.manual_seed(42)

    model_xavier = DeepNeuralNetwork()

    initialize_xavier(model_xavier)

    # --------------------------------------------------------
    # HE MODEL
    # --------------------------------------------------------

    torch.manual_seed(42)

    model_he = DeepNeuralNetwork()

    initialize_he(model_he)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_model(
        model_default,
        dataloader,
        "Default Initialization"
    )

    train_model(
        model_xavier,
        dataloader,
        "Xavier Initialization"
    )

    train_model(
        model_he,
        dataloader,
        "He Initialization"
    )

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    evaluate_model(
        model_default,
        "Default Initialization"
    )

    evaluate_model(
        model_xavier,
        "Xavier Initialization"
    )

    evaluate_model(
        model_he,
        "He Initialization"
    )

b
if __name__ == "__main__":
    main()