import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as functional
import numpy as np
from sklearn.metrics import roc_auc_score
import sys

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on the GPU")
else:
    device = torch.device("cpu")
    print("running on the CPU")

num_epochs = 15
num_classes = 10
batch_size = 128
learning_rate = 0.001

train_dataset = torchvision.datasets.MNIST(root='../../data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='../../data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dense1 = nn.Linear(12 * 12 * 64, 128)
        self.dense2 = nn.Linear(128, num_classes)

    def forward(self, x):
        print(x[0])
        print(x[0].shape)
        print("---------------------------------------")
        x = functional.relu(self.conv1(x))
        print(self.conv1.weight)
        print(self.conv1.weight.shape)
        print("---------------------------------------")
        print(x[0])
        print(x[0].shape)
        exit()
        x = functional.relu(self.conv2(x))
        x = functional.max_pool2d(x, 2, 2)
        x = functional.dropout(x, 0.25)
        x = x.view(-1, 12 * 12 * 64)
        x = functional.relu(self.dense1(x))
        x = functional.dropout(x, 0.5)
        y = functional.log_softmax(self.dense2(x), dim=1)
        return y


model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, i+1, total_step, loss.item()))


preds = []
y_true = []
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        detached_pred = predicted.detach().cpu().numpy()
        detached_label = labels.detach().cpu().numpy()
        for f in range(0, len(detached_pred)):
            preds.append(detached_pred[f])
            y_true.append((detached_label[f]))

    print("Test Accuracy of the model on 1000 test images: {:.2%}".format(correct / total))
    preds = np.eye(num_classes)[preds]
    y_true = np.eye(num_classes)[y_true]
    auc = roc_auc_score(preds, y_true)
    print("AUC: {:.2%}".format(auc))

torch.save(model.state_dict(), 'pytorch_mnist_cnn.ckpt')
