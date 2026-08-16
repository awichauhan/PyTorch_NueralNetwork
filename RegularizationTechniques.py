import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. CREATE SIMPLE BINARY CLASSIFICATION DATA
# ============================================================

torch.manual_seed(42)

X = torch.randn(200, 2)
# Given two numbers, predict whether their sum is positive or not.
Y = (
    (X[:, 0] + X[:, 1]) > 0
).float().reshape(-1, 1)


# ============================================================
# 2. TRAIN / VALIDATION SPLIT
# ============================================================

X_train = X[:160]
Y_train = Y[:160]

X_val = X[160:]
Y_val = Y[160:]


# ============================================================
# 3. DATASET
# ============================================================

class BinaryDataset(Dataset):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


# ============================================================
# 4. CREATE DATASETS AND DATALOADERS
# ============================================================

train_dataset = BinaryDataset(
    X_train,
    Y_train
)

val_dataset = BinaryDataset(
    X_val,
    Y_val
)


train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)


# ============================================================
# 5. BASE MODEL
# ============================================================

class NeuralNetwork(nn.Module):

    def __init__(self, use_dropout=False):

        super().__init__()

        self.layer1 = nn.Linear(2, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, 16)
        self.output_layer = nn.Linear(16, 1)

        self.use_dropout = use_dropout

        self.dropout = nn.Dropout(
            p=0.3
        )


    def forward(self, X):

        Z1 = self.layer1(X)
        A1 = torch.relu(Z1)

        if self.use_dropout:
            A1 = self.dropout(A1)


        Z2 = self.layer2(A1)
        A2 = torch.relu(Z2)

        if self.use_dropout:
            A2 = self.dropout(A2)


        Z3 = self.layer3(A2)
        A3 = torch.relu(Z3)


        Z4 = self.output_layer(A3)

        return Z4


# ============================================================
# 6. HE INITIALIZATION
# ============================================================

def initialize_he(model):

    for layer in model.modules():

        if isinstance(layer, nn.Linear):

            nn.init.kaiming_uniform_(
                layer.weight,
                nonlinearity="relu"
            )

            nn.init.zeros_(
                layer.bias
            )


# ============================================================
# 7. VALIDATION FUNCTION
# ============================================================

def validate_model(model, criterion):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for X_batch, Y_batch in val_loader:

            logits = model(X_batch)

            loss = criterion(
                logits,
                Y_batch
            )

            total_loss += loss.item()

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= 0.5
            ).float()

            correct += (
                predictions == Y_batch
            ).sum().item()

            total += Y_batch.size(0)

    average_loss = (
        total_loss / len(val_loader)
    )

    accuracy = (
        correct / total
    )

    return average_loss, accuracy


# ============================================================
# 8. TRAINING FUNCTION
# ============================================================

def train_model(
        model,
        model_name,
        weight_decay=0.0,
        use_early_stopping=False
):

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.05,
        weight_decay=weight_decay
    )

    epochs = 500

    best_val_loss = float("inf")

    patience = 20

    patience_counter = 0


    print(
        "\nTraining:",
        model_name
    )


    for epoch in range(epochs):

        # --------------------------------
        # TRAINING MODE
        # --------------------------------

        model.train()

        total_train_loss = 0


        for X_batch, Y_batch in train_loader:

            logits = model(
                X_batch
            )

            loss = criterion(
                logits,
                Y_batch
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_train_loss += (
                loss.item()
            )


        average_train_loss = (
            total_train_loss /
            len(train_loader)
        )


        # --------------------------------
        # VALIDATION
        # --------------------------------

        val_loss, val_accuracy = (
            validate_model(
                model,
                criterion
            )
        )


        # --------------------------------
        # PRINT PROGRESS
        # --------------------------------

        if epoch % 50 == 0:

            print(
                "Epoch:",
                epoch,
                "| Train Loss:",
                round(
                    average_train_loss,
                    4
                ),
                "| Val Loss:",
                round(
                    val_loss,
                    4
                ),
                "| Val Accuracy:",
                round(
                    val_accuracy,
                    4
                )
            )


        # --------------------------------
        # EARLY STOPPING
        # --------------------------------

        if use_early_stopping:

            if val_loss < best_val_loss:

                best_val_loss = val_loss

                patience_counter = 0

            else:

                patience_counter += 1


            if patience_counter >= patience:

                print(
                    "Early stopping at epoch:",
                    epoch
                )

                break


    return model


# ============================================================
# 9. FINAL EVALUATION
# ============================================================

def final_evaluation(
        model,
        model_name
):

    criterion = nn.BCEWithLogitsLoss()

    val_loss, val_accuracy = (
        validate_model(
            model,
            criterion
        )
    )

    print(
        "\nFinal Result:",
        model_name
    )

    print(
        "Validation Loss:",
        round(
            val_loss,
            4
        )
    )

    print(
        "Validation Accuracy:",
        round(
            val_accuracy,
            4
        )
    )


# ============================================================
# 10. MAIN
# ============================================================

def main():

    # ========================================================
    # BASELINE MODEL
    # ========================================================

    torch.manual_seed(42)

    baseline_model = NeuralNetwork()

    initialize_he(
        baseline_model
    )

    train_model(
        baseline_model,
        "Baseline"
    )


    # ========================================================
    # L2 MODEL
    # ========================================================

    torch.manual_seed(42)

    l2_model = NeuralNetwork()

    initialize_he(
        l2_model
    )

    train_model(
        l2_model,
        "L2 Regularization",
        weight_decay=0.001
    )


    # ========================================================
    # DROPOUT MODEL
    # ========================================================

    torch.manual_seed(42)

    dropout_model = NeuralNetwork(
        use_dropout=True
    )

    initialize_he(
        dropout_model
    )

    train_model(
        dropout_model,
        "Dropout"
    )


    # ========================================================
    # EARLY STOPPING MODEL
    # ========================================================

    torch.manual_seed(42)

    early_stop_model = NeuralNetwork()

    initialize_he(
        early_stop_model
    )

    train_model(
        early_stop_model,
        "Early Stopping",
        use_early_stopping=True
    )


    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    final_evaluation(
        baseline_model,
        "Baseline"
    )

    final_evaluation(
        l2_model,
        "L2 Regularization"
    )

    final_evaluation(
        dropout_model,
        "Dropout"
    )

    final_evaluation(
        early_stop_model,
        "Early Stopping"
    )


if __name__ == "__main__":
    main()