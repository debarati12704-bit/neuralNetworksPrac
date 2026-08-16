import torch
import torch.nn as nn
import torch.optim as optim


# Dataset
X = torch.tensor([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=torch.float32)

y = torch.tensor([
    [0],
    [1],
    [1],
    [0]
], dtype=torch.float32)


# Neural Network
class XORNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(2, 4)
        self.output = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)

        return x


# Create model
model = XORNetwork()


# Loss function
criterion = nn.BCEWithLogitsLoss()


# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)


# Training
for epoch in range(5000):

    # Forward propagation
    predictions = model(X)

    # Loss
    loss = criterion(predictions, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()

    # Update parameters
    optimizer.step()


# Testing
with torch.no_grad():

    output = model(X)

    probabilities = torch.sigmoid(output)

    predictions = (probabilities >= 0.5).float()

    print("Probabilities:")
    print(probabilities)

    print("\nPredictions:")
    print(predictions)