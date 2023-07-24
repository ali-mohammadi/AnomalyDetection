import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os
import cv2
import numpy as np
import sys

mnist_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
mnist_data = list(mnist_data)[:4096]

class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(4, 8, 4, 1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 4, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 8, 1),
            nn.ReLU(),
            nn.Conv2d(32, 128, 16),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 16),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 8, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 4, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, 2, 1),
            nn.Tanh(),
        )

        # self.encoder = nn.Sequential(  # like the Composition layer you built
        #     nn.Conv2d(1, 16, 3, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, 3, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, 7)
        # )
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(64, 32, 7),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train(model, dataset, num_epoch=5, batch_size=64, learning_rate=1e-3):
    torch.manual_seed(42)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-5)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # train_loader = torch.utils.data.DataLoader(mnist_data,
    #                                            batch_size=batch_size,
    #                                            shuffle=True)
    outputs = []
    for epoch in range(num_epoch):
        for data in train_loader:
            img, _ = data
            recon = model(img)
            loss = criterion(recon, img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
        outputs.append((epoch, img, recon), )
    return outputs


def main():
    instances = []
    for filepath in os.listdir('packet_images/'):
        instances.append(np.expand_dims(cv2.imread('packet_images/{0}'.format(filepath), 0), axis=0))
    instances = np.stack(instances)
    dataset = MyDataset(instances, np.zeros(len(instances)))
    model = Autoencoder()
    outputs = train(model, dataset, 20, learning_rate=1e-5)

    # for k in range(0, 20, 1):
    k = 19
    plt.figure(figsize=(9, 2))
    imgs = outputs[k][1].detach().numpy()
    recon = outputs[k][2].detach().numpy()
    # for i, item in enumerate(imgs):
    #     cv2.imshow('img', item[0])
    #     # cv2.waitKey(0)
    #     continue
    #     if i >= 9: break
    #     plt.subplot(2, 9, i + 1)
    #     plt.imshow(item[0])

    for i, item in enumerate(recon):
        cv2.imshow('recon', item[0])
        cv2.waitKey(0)
        continue
        if i >= 9: break
        plt.subplot(2, 9, 9 + i + 1)
        plt.imshow(item[0])

main()
