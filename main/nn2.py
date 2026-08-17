import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset,DataLoader

#data
X = torch.tensor([
    [1.0, 0.0],
    [1.5, 1.0],
    [2.0, 0.0],
    [2.0, 1.0],
    [2.5, 1.0],
    [3.0, 1.0],
    [3.0, 2.0],
    [3.5, 2.0],
    [4.0, 2.0],
    [4.0, 3.0],
    [4.5, 3.0],
    [5.0, 3.0],
    [5.0, 4.0],
    [6.0, 4.0],
    [6.0, 5.0],
    [7.0, 5.0]
], dtype=torch.float32)

y = torch.tensor([
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1]
], dtype=torch.float32)

class myDataset(Dataset):
    def __init__(self,X,y):
        self.X=X
        self.y=y
    def __len__(self):
        return len(self.X)
    def __getitem__(self,index):
        return self.X[index],self.y[index]

class myModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(2,8)
        self.layer2=nn.Linear(8,4)
        self.layer3=nn.Linear(4,1)
    def forward(self,x):
        x=torch.relu(self.layer1(x))
        x=torch.relu(self.layer2(x))
        x=self.layer3(x)
        return x

dataset=myDataset(X,y)

loader=DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)
model=myModel()
criterion=nn.BCEWithLogitsLoss()
optimizer=optim.Adam(
    model.parameters(),
    lr=0.1
)
for epoch in range(1750):
    model.train()
    for X_batch,y_batch in loader:
        output=model(X_batch)
        loss=criterion(output,y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    output=model(X)
    probabilities=torch.sigmoid(output)
    predictions=( probabilities>=0.5 ).float()
    print("\nOutput:",output)
    print("\nProbabilities:",probabilities)
    print("\nPredictions:",predictions)

