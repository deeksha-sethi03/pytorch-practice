'''
WAP to implement linear regression using PyTorch. Make a dummy dataset for the task.
'''
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
def createDataset(size = 100):
    X = torch.rand(size, 1)
    # noise = torch.randn(size, 1) # -> this will lead to imperfect fitting and the loss will never converge to 0.
    y = 0.23 * X + 0.87 # + noise
    return X, y

class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.LinearRegressor = nn.Linear(1, 1)

    def forward(self, X):
        return self.LinearRegressor(X)
    
size = 200
X, y_true = createDataset(size)

model = LinearRegression()
optimizer = optim.Adam(model.parameters(), lr = 0.01)
criterion = nn.MSELoss()

epochs = 1000

for i in range(epochs):
    y_pred = model(X)
    loss = criterion(y_pred, y_true)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if(i%100 == 0):
        print("Epoch : {} | Loss : {}".format(i, loss))

[w, b] = model.LinearRegressor.parameters()
print(w, b)

with torch.no_grad():
    y_pred = model(X[0])
    print(y_pred, y_true[0])





