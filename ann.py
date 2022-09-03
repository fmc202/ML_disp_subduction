import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F


class GroundMotionDataset(Dataset):
    def __init__(self, x, y):
        self.num_samples = x.size()[0]
        self.data = x
        self.labels = y  # y must be the shape of (num_samples, 1) both x y must be tensor

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        data = self.data[idx, :]
        label = self.labels[idx, :]

        return data, label


def train_loop(dataloader, model, loss_fn, optimizer):
    epoch_loss = 0
    num_sample = 0

    for _, (X, y) in enumerate(dataloader):
        pred = model(X)  # input x and predict based on x
        loss = loss_fn(pred, y)
        batchsize = X.size()[0]

        optimizer.zero_grad()  # clear gradients
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        epoch_loss += loss.item() * batchsize
        num_sample += batchsize

    epoch_loss /= num_sample
    print(f"training epoch loss is {epoch_loss:.5f}")

    return epoch_loss


def val_loop(dataloader, model, loss_fn):
    num_sample = 0
    loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            batchsize = X.size()[0]
            pred = model(X)
            loss += loss_fn(pred, y).item() * batchsize
            num_sample += batchsize

    loss /= num_sample  # average loss per batch to be comparable with training loss
    print(f"validation epoch loss is {loss:.5f}")
    return loss


class ResNet(nn.Module):
    def __init__(self, n_feature, n_output):

        super(ResNet, self).__init__()
        self.h1 = nn.Sequential(
            nn.Linear(n_feature, 1000),
            nn.BatchNorm1d(1000),
        )
        self.r1 = nn.Linear(n_feature, 1000)

        self.h2 = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
        )
        self.h3 = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
        )
        self.h4 = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
        )
        self.h5 = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
        )
        self.h6 = nn.Sequential(
            nn.Linear(1000, 500),
            nn.BatchNorm1d(500),
        )
        self.r6 = nn.Linear(1000, 500)

        self.h7 = nn.Sequential(
            nn.Linear(500, 500),
            nn.BatchNorm1d(500),
        )
        self.h8 = nn.Sequential(
            nn.Linear(500, 500),
            nn.BatchNorm1d(500),
        )
        self.h9 = nn.Sequential(
            nn.Linear(500, 500),
            nn.BatchNorm1d(500),
        )
        self.h10 = nn.Sequential(
            nn.Linear(500, 500),
            nn.BatchNorm1d(500),
        )
        self.h11 = nn.Sequential(
            nn.Linear(500, 250),
            nn.BatchNorm1d(250),
        )
        self.r11 = nn.Linear(500, 250)

        self.h12 = nn.Sequential(
            nn.Linear(250, 250),
            nn.BatchNorm1d(250),
        )
        self.h13 = nn.Sequential(
            nn.Linear(250, 250),
            nn.BatchNorm1d(250),
        )
        self.h14 = nn.Sequential(
            nn.Linear(250, 250),
            nn.BatchNorm1d(250),
        )
        self.h15 = nn.Sequential(
            nn.Linear(250, 250),
            nn.BatchNorm1d(250),
        )
        self.h16 = nn.Sequential(
            nn.Linear(250, 125),
            nn.BatchNorm1d(125),
        )
        self.r16 = nn.Linear(250, 125)

        self.h17 = nn.Sequential(
            nn.Linear(125, 125),
            nn.BatchNorm1d(125),
        )
        self.h18 = nn.Sequential(
            nn.Linear(125, 125),
            nn.BatchNorm1d(125),
        )
        self.h19 = nn.Sequential(
            nn.Linear(125, 125),
            nn.BatchNorm1d(125),
        )
        self.h20 = nn.Sequential(
            nn.Linear(125, 125),
            nn.BatchNorm1d(125),
        )
        self.h21 = nn.Sequential(
            nn.Linear(125, 64),
            nn.BatchNorm1d(64),
        )
        self.r21 = nn.Linear(125, 64)

        self.h22 = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
        )
        self.h23 = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
        )
        self.h24 = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
        )
        self.h25 = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
        )
        self.h26 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
        )
        self.r26 = nn.Linear(64, 32)

        self.h27 = nn.Sequential(
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
        )
        self.h28 = nn.Sequential(
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
        )

        self.h29 = nn.Sequential(
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
        )

        self.h30 = nn.Sequential(
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
        )

        self.h31 = nn.Sequential(
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
        )

        self.predict = nn.Linear(32, n_output)  # output layer

    def forward(self, x):
        out = F.relu(self.h1(x) + self.r1(x))
        out = F.relu(self.h2(out) + out)
        out = F.relu(self.h3(out) + out)
        out = F.relu(self.h4(out) + out)
        out = F.relu(self.h5(out) + out)
        out = F.relu(self.h6(out) + self.r6(out))
        out = F.relu(self.h7(out) + out)
        out = F.relu(self.h8(out) + out)
        out = F.relu(self.h9(out) + out)
        out = F.relu(self.h10(out) + out)
        out = F.relu(self.h11(out) + self.r11(out))
        out = F.relu(self.h12(out) + out)
        out = F.relu(self.h13(out) + out)
        out = F.relu(self.h14(out) + out)
        out = F.relu(self.h15(out) + out)
        out = F.relu(self.h16(out) + self.r16(out))
        out = F.relu(self.h17(out) + out)
        out = F.relu(self.h18(out) + out)
        out = F.relu(self.h19(out) + out)
        out = F.relu(self.h20(out) + out)
        out = F.relu(self.h21(out) + self.r21(out))
        out = F.relu(self.h22(out) + out)
        out = F.relu(self.h23(out) + out)
        out = F.relu(self.h24(out) + out)
        out = F.relu(self.h25(out) + out)
        out = F.relu(self.h26(out) + self.r26(out))
        out = F.relu(self.h27(out) + out)
        out = F.relu(self.h28(out) + out)
        out = F.relu(self.h29(out) + out)
        out = F.relu(self.h30(out) + out)
        out = F.relu(self.h31(out) + out)

        out = self.predict(out)
        return out
