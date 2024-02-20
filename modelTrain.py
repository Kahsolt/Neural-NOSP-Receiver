# =======================================================================================================================
# =======================================================================================================================
import os
import numpy as np
import h5py

from modelDesign import *

# Parameters Setting
# ========================================================
## case1
NUM_SUBCARRIERS = 624
NUM_OFDM_SYMBOLS = 12
NUM_LAYERS = 2
NUM_BITS_PER_SYMBOL = 4

## case2
# NUM_SUBCARRIERS = 96
# NUM_OFDM_SYMBOLS = 12
# NUM_LAYERS = 4
# NUM_BITS_PER_SYMBOL = 6


EPOCHS = 2
train_dataset_dir = './../data/'  # Please set parameters according to your local path of data

# Data Loading
# ========================================================
print('=====================load case1 data===============')
f = h5py.File(os.path.join(train_dataset_dir, "D1.hdf5"), 'r')
rx_signal = f['rx_signal'][:]
tx_bits = f['tx_bits'][:]
pilot = f['pilot'][:]
f.close()
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


def generator(batch, rx_signal_in, pilot_in, tx_bits_in):
    idx_tmp = np.random.choice(rx_signal_in.shape[0], batch, replace=False)
    batch_rx_signal = rx_signal_in[idx_tmp].astype(np.float32)
    batch_pilot = pilot_in[idx_tmp].astype(np.float32)
    batch_tx_bits = tx_bits_in[idx_tmp].astype(np.float32)
    return torch.from_numpy(batch_rx_signal), torch.from_numpy(batch_pilot), torch.from_numpy(batch_tx_bits)


# Model Constructing
# ========================================================
Model = Neural_receiver(subcarriers=NUM_SUBCARRIERS,
                        timesymbols=NUM_OFDM_SYMBOLS, streams=NUM_LAYERS,
                        num_bits_per_symbol=NUM_BITS_PER_SYMBOL)
Model = Model.cuda()
criterion = nn.BCEWithLogitsLoss().cuda()

optimizer = torch.optim.Adam(Model.parameters(), lr=1e-3)

# Model Training and Saving
# =========================================================
bestLoss = 100
for epoch in range(EPOCHS):
    Model.train()
    ModelInput1, ModelInput2, label = generator(16, rx_signal_train, pilot_train, tx_bits_train)
    ModelInput1, ModelInput2, label = ModelInput1.cuda(), ModelInput2.cuda(), label.cuda()
    ModelOutput = Model(ModelInput1, ModelInput2)
    loss = criterion(ModelOutput, label)
    predict = torch.where(ModelOutput >= 0, 1.0, 0.0)
    score = torch.where(predict == label, 1.0, 0.0)
    acc = torch.mean(score)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Model Evaluating
    Model.eval()
    with torch.no_grad():
        val_ModelInput1, val_ModelInput2, val_label = generator(16, rx_signal_val, pilot_val, tx_bits_val)
        val_ModelInput1, val_ModelInput2, val_label = val_ModelInput1.cuda(), val_ModelInput2.cuda(), val_label.cuda()
        val_ModelOutput = Model(val_ModelInput1, val_ModelInput2)
        val_loss = criterion(val_ModelOutput, val_label).item()
        val_predict = torch.where(val_ModelOutput >= 0, 1.0, 0.0)
        val_score = torch.where(val_predict == val_label, 1.0, 0.0)
        val_acc = torch.mean(val_score)
        print(
            'Epoch: [{0}]\t' 'Loss {loss:.4f}\t' 'Acc {acc:.4f}\t' 'val_loss {val_loss:.4f}\t' 'val_acc {val_acc:.4f}\t'.format(
                epoch, loss=loss.item(), acc=acc, val_loss=val_loss, val_acc=val_acc))
        if val_loss < bestLoss:
            # Model saving
            torch.save(Model, 'receiver_1b.pth.tar')
            print("Model saved")
            bestLoss = val_loss
print('Training for case_1b is finished!')
