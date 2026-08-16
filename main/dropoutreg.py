import torch
from torch import nn

# Model
model = nn.Sequential(
    nn.Linear(3, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 10)
)


# Input
X = torch.tensor([
    [1.2, 2.4, 1.7],
    [2.4, 3.6, 2.9]
], dtype=torch.float32)


# Training mode
model.train()

output = model(X)

print("Output:")
print(output)
print(model.eval())
