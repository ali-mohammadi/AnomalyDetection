import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cpu')


class UCI_Digits_Dataset(torch.utils.data.Dataset):
    # 8,12,0,16, . . 15,7
    # 64 pixel values [0-16], digit [0-9]

    def __init__(self, src_file, n_rows=None):
        all_xy = np.loadtxt(src_file, max_rows=n_rows,
                            usecols=range(0, 65), delimiter=",", comments="#",
                            dtype=np.float32)
        self.xy_data = torch.tensor(all_xy, dtype=torch.float32).to(device)
        self.xy_data[:, 0:64] /= 16.0  # normalize pixels
        self.xy_data[:, 64] /= 9.0  # normalize digit/label

    def __len__(self):
        return len(self.xy_data)

    def __getitem__(self, idx):
        xy = self.xy_data[idx]
        return xy


def display_digit(ds, idx, save=False):
    # ds is a PyTorch Dataset
    line = ds[idx]  # tensor
    pixels = np.array(line[0:64])  # numpy row of pixels
    label = np.int(line[64] * 9.0)  # denormalize; like '5'
    print("\ndigit = ", str(label), "\n")

    pixels = pixels.reshape((8, 8))
    for i in range(8):
        for j in range(8):
            pxl = pixels[i, j]  # or [i][j] syntax
            pxl = np.int(pxl * 16.0)  # denormalize
            print("%.2X" % pxl, end="")
            print(" ", end="")
        print("")

    plt.imshow(pixels, cmap=plt.get_cmap('gray_r'))
    if save == True:
        plt.savefig(".\\idx_" + str(idx) + "_digit_" + \
                    str(label) + ".jpg", bbox_inches='tight')
    plt.show()
    plt.close()


def display_digits(ds, idxs, save=False):
    # idxs is a list of indices
    for idx in idxs:
        display_digit(ds, idx, save)


class Autoencoder(torch.nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.layer1 = torch.nn.Linear(65, 32)
        self.layer2 = torch.nn.Linear(32, 8)
        self.layer3 = torch.nn.Linear(8, 32)
        self.layer4 = torch.nn.Linear(32, 65)

    def encode(self, x):
        z = torch.tanh(self.layer1(x))
        z = torch.tanh(self.layer2(z))
        return z

    def decode(self, x):
        z = torch.tanh(self.layer3(x))
        z = torch.sigmoid(self.layer4(z))
        return z

    def forward(self, x):
        z = self.encode(x)
        z = self.decode(z)
        return z


def train(model, dataset, batch_size, max_epoch, log_interval, learning_rate):
    ldr = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_func = torch.nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("\nStart training")
    for epoch in range(0, max_epoch):
        epoch_loss = 0.0
        for (batch_index, batch) in enumerate(ldr):
            X = batch
            Y = batch

            optimizer.zero_grad()
            output = model(X)
            loss_value = loss_func(output, Y)
            epoch_loss += loss_value.item()
            loss_value.backward()
            optimizer.step()

        if epoch % log_interval == 0:
            print("epoch = %4d   loss = %0.4f" % (epoch, epoch_loss))
    print("Done")


def make_err_list(model, dataset):
    result_list = []
    n_features = len(dataset[0])
    for i in range(len(dataset)):
        X = dataset[i]
        with torch.no_grad():
            Y = model(X)
        err = torch.sum((X - Y) * (X - Y)).item()
        err = err / n_features
        result_list.append((i, err))
    return result_list


def main():
    torch.manual_seed(1)
    np.random.seed(1)
    fn = "./optdigits.tra"
    ds = UCI_Digits_Dataset(fn)

    autoenc = Autoencoder().to(device)
    autoenc.train()

    bat_size = 10
    max_epochs = 100
    log_interval = 10
    lrn_rate = 0.005

    print("bat_size = %3d " % bat_size)
    print("max epochs = " + str(max_epochs))
    print("loss = MSELoss")
    print("optimizer = SGD")
    print("lrn_rate = %0.3f " % lrn_rate)
    train(autoenc, ds, bat_size, max_epochs, log_interval, lrn_rate)

    print("Computing reconstruction errors ")
    autoenc.eval()  # set mode
    err_list = make_err_list(autoenc, ds)
    err_list.sort(key=lambda x: x[1], reverse=True)

    print("Largest reconstruction item / error: ")
    (idx, err) = err_list[0]
    print(" [%4d]  %0.4f" % (idx, err))
    display_digit(ds, idx)
    print("\nEnd autoencoder anomaly detection demo \n")


if __name__ == "__main__":
    main()
    # fn = "./optdigits.tra"
    # ds = UCI_Digits_Dataset(fn)
