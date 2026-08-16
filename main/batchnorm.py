import torch
from torch import nn


# Model
model = nn.Sequential(
    nn.Linear(3, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Linear(128, 1)
)


# Input
X = torch.tensor([
    [1.2, 2.4, 1.7],
    [2.4, 3.6, 2.9],
    [3.1, 4.2, 3.5],
    [4.0, 5.1, 4.2]
], dtype=torch.float32)


# Forward propagation
model.train()

output = model(X)

print("Input shape:", X.shape)
print("Output shape:", output.shape)
print("Output:")
print(output)