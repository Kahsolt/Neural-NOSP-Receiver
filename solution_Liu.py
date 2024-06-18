import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch import Tensor
import h5py
import numpy as np

NUM_SUBCARRIERS = 624
NUM_OFDM_SYMBOLS = 12
NUM_LAYERS = 2
NUM_BITS_PER_SYMBOL = 4

EPOCHS = 1000

device = 'cuda'
train_dataset_dir = './data/' 


def get_data():
    with h5py.File(os.path.join(train_dataset_dir, "D1.hdf5"), 'r') as f:
        rx_signal = np.array(f['rx_signal'][:])
        tx_bits   = np.array(f['tx_bits']  [:])
        pilot     = np.array(f['pilot']    [:])
    print('rx_signal:', rx_signal.shape, rx_signal.dtype)
    print('tx_bits:', tx_bits.shape, tx_bits.dtype)
    samples = rx_signal.shape[0]
    pilot = np.tile(pilot, [samples, 1, 1, 1, 1])
    print('pilot:', pilot.shape, pilot.dtype)

    rx_signal_train = rx_signal[:int(rx_signal.shape[0] * 0.99)]
    rx_signal_val = rx_signal[int(rx_signal.shape[0] * 0.99):]
    pilot_train = pilot[:int(pilot.shape[0] * 0.99)]
    pilot_val = pilot[int(pilot.shape[0] * 0.99):]
    tx_bits_train = tx_bits[:int(tx_bits.shape[0] * 0.99)]
    tx_bits_val = tx_bits[int(tx_bits.shape[0] * 0.99):]

    print('rx_signal_train:', rx_signal_train.shape, rx_signal_train.dtype)
    print('tx_bits_train:', tx_bits_train.shape, tx_bits_train.dtype)
    print('pilot_train:', pilot_train.shape, pilot_train.dtype)

    print('rx_signal_val:', rx_signal_val.shape, rx_signal_val.dtype)
    print('tx_bits_val:', tx_bits_val.shape, tx_bits_val.dtype)
    print('pilot_val:', pilot_val.shape, pilot_val.dtype)

    w = 0.1 # 权重
    data  = np.concatenate((rx_signal_train, pilot_train * w), axis=4)
    label = tx_bits_train[..., 0] * 1 + tx_bits_train[..., 1] * 2 + tx_bits_train[..., 2] * 4 + tx_bits_train[..., 3] * 8
    label = label.astype(np.int32)

    data  = data .reshape(-1, 624, 4)       # [B*L*T, S, c*2]
    label = label.reshape(-1, 624, 1)
    label = label[:, 0, 0]          # select the first subcarier bit
    print('data:', data.shape, data.dtype)
    print('label:', label.shape, label.dtype)
    return data, label


def generator(batch_size:int, data:np.ndarray, label:np.ndarray):
    idx = np.random.choice(data.shape[0], batch_size, replace=False)
    batch_data  = data [idx].astype(np.float32)
    batch_label = label[idx].astype(np.int32)
    return torch.from_numpy(batch_data).float(), torch.from_numpy(batch_label).long()


class NeuralNetwork(nn.Module):

    # Modeling: S subcariers [B, D=S*c*2] -> the first code bit [B, M=16]

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(4 * 624, 900)
        self.fc2 = nn.Linear(900, 450)
        self.fc3 = nn.Linear(450, 240)
        self.fc4 = nn.Linear(240, 120)
        self.fc5 = nn.Linear(120, 64)
        self.fc6 = nn.Linear(64, 16)

    def forward(self, x:Tensor) -> Tensor:
        # x.shape: [B, S, D=c*2]
        x = x.view(x.shape[0], -1)      # [B, S*c*2]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        return x


if __name__ == '__main__':
    data, label = get_data()
    model = NeuralNetwork().to(device)
    print(model)
    optim = Adam(model.parameters(), lr=0.1)

    model.train()
    for epoch in range(EPOCHS):
        optim.zero_grad()
        X, Y = generator(4, data, label)
        X, Y = X.to(device), Y.to(device)
        outputs = model(X)
        loss = F.cross_entropy(outputs, Y)
        loss.backward()
        optim.step()

        if epoch % 100 == 0:
            acc = (outputs.argmax(-1) == Y).sum().item() / len(Y)
            print(f"Epoch {epoch+1}/{EPOCHS}, loss: {loss.item()}, acc: {acc}")
