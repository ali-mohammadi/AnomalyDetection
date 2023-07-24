import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

device = torch.device('cpu')

class autoencoder(torch.nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.layer1 = torch.nn.Linear(29, 16)
        self.layer2 = torch.nn.Linear(16, 8)
        self.layer3 = torch.nn.Linear(8, 16)
        self.layer4 = torch.nn.Linear(16, 29)

    def encode(self, x):
        z = torch.tanh(self.layer1(x))
        leaky_relu = torch.nn.LeakyReLU()
        z = leaky_relu(self.layer2(z))
        return z

    def decode(self, x):
        z = torch.tanh(self.layer3(x))
        leaky_relu = torch.nn.LeakyReLU()
        z = leaky_relu(self.layer4(z))
        return z

    def forward(self, x):
        z = self.encode(x)
        z = self.decode(z)
        return z


def train(model, dataset, batch_size, max_epochs, learning_rate):
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_fun = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(0, max_epochs):
        epoch_loss = 0.0
        for batch in dataset_loader:
            X = batch.float()
            Y = batch.float()

            optim.zero_grad()
            Z = model(X)
            loss_val = loss_fun(Z, Y)
            epoch_loss += loss_val.item()
            loss_val.backward()
            optim.step()
        if epoch % 10 == 0:
            print("Epoch = {:4d}, Loss = {:.4f}".format(epoch, epoch_loss))

ds = pd.read_csv('./creditcard.csv')
label = ds.iloc[:, -1:]
# dataset = ds.drop(['Class'], axis=1)
data = ds.drop(['Class', 'Time'], axis=1)

data['Amount'] = StandardScaler().fit_transform(data["Amount"].values.reshape(-1, 1))

x_train, x_test, y_train, y_test = train_test_split(data.values, label, test_size=0.2, random_state=42)
autoenc = autoencoder().to(device)
autoenc.train()
train(autoenc, x_train, 256, 100, 0.01)
torch.save(autoenc.state_dict(), './autoenc.pt')
#
# autoenc.eval()
# for i in range(len(x_test)):
#     X = x_test[i]
#     with torch.no_grad():
#         Y = autoenc(X)
#     loss = torch.sum((X - Y) * (X - Y)).item()
#     print(loss)