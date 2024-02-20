import os
import time
import numpy as np
import h5py
from torch.utils.data import Dataset

from modelDesign import *


# Data Loader Class Defining
class DatasetFolder_eval(Dataset):
    def __init__(self, y, p):
        self.y = y.astype(np.float32)
        self.p = p.astype(np.float32)

    def __getitem__(self, index):
        return torch.tensor(self.y[index]), torch.tensor(self.p[index])

    def __len__(self):
        return len(self.y)


# ======================================================================================================================
test_data_dir = './../data_test_1W/'
model_dir = './submit_pt/modelSubmit/'

# ======================================================================================================================
# Data Loading
t0 = time.time()
print('=====================load case1 test data===============')
f = h5py.File(os.path.join(test_data_dir, "test_data_case1.hdf5"), 'r')
rx_signal_test_1 = f['rx_signal'][:]
pilot_1 = f['pilot'][:]
tx_bits_test_1 = f['tx_bits'][:]
f.close()
print('rx_signal_test:', rx_signal_test_1.shape, rx_signal_test_1.dtype)
print('tx_bits_test:', tx_bits_test_1.shape, tx_bits_test_1.dtype)
samples = rx_signal_test_1.shape[0]
pilot_1 = np.tile(pilot_1, [samples, 1, 1, 1, 1])
print('pilot:', pilot_1.shape, pilot_1.dtype)


print('=====================load case2 test data===============')
f = h5py.File(os.path.join(test_data_dir, "test_data_case2.hdf5"), 'r')
rx_signal_test_2 = f['rx_signal'][:]
pilot_2 = f['pilot'][:]
tx_bits_test_2 = f['tx_bits'][:]
f.close()
print('rx_signal_test:', rx_signal_test_2.shape, rx_signal_test_2.dtype)
print('tx_bits_test:', tx_bits_test_2.shape, tx_bits_test_2.dtype)
samples = rx_signal_test_2.shape[0]
pilot_2 = np.tile(pilot_2, [samples, 1, 1, 1, 1])
print('pilot:', pilot_2.shape, pilot_2.dtype)

t1 = time.time()
print('load data：{}s'.format(t1 - t0))

tx_bits_test_1 = torch.Tensor(tx_bits_test_1).cuda()
tx_bits_test_2 = torch.Tensor(tx_bits_test_2).cuda()

# ======================================================================================================================
# Load Model & Interfere

print('===================== case_1a ==========================')
model = torch.load(os.path.join(model_dir, 'receiver_1a.pth.tar')).cuda()
test_dataset = DatasetFolder_eval(rx_signal_test_1, pilot_1)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
model.eval()

t2 = time.time()
print("load model：{}s".format(t2 - t1))
output_all = []
with torch.no_grad():
    for idx, data in enumerate(test_loader):
        y, p = data
        y, p = y.cuda(), p.cuda()
        modelOutput = model(y, p)
        output_all.append(modelOutput)
output_all = torch.cat(output_all, dim=0)

t3 = time.time()
print("model eval：{}s".format(t3 - t2))
predict = torch.where(output_all >= 0, 1.0, 0.0)
score = torch.where(predict == tx_bits_test_1, 1.0, 0.0)
acc_1a = score.mean().cpu().numpy()
t4 = time.time()
print("count score：{}s".format(t4 - t3))
print('case_1a: acc= ' + str(acc_1a))

print('===================== case_1b ==========================')
model = torch.load(os.path.join(model_dir, 'receiver_1b.pth.tar')).cuda()
test_dataset = DatasetFolder_eval(rx_signal_test_1, pilot_1)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
model.eval()

t5 = time.time()
print("load model：{}s".format(t5 - t4))
output_all = []
with torch.no_grad():
    for idx, data in enumerate(test_loader):
        y, p = data
        y, p = y.cuda(), p.cuda()
        modelOutput = model(y, p)
        output_all.append(modelOutput)
output_all = torch.cat(output_all, dim=0)

t6 = time.time()
print("model eval：{}s".format(t6 - t5))
predict = torch.where(output_all >= 0, 1.0, 0.0)
score = torch.where(predict == tx_bits_test_1, 1.0, 0.0)
acc_1b = score.mean().cpu().numpy()
t7 = time.time()
print("count score：{}s".format(t7 - t6))
print('case_1b: acc= ' + str(acc_1b))

print('===================== case2 ==========================')
model = torch.load(os.path.join(model_dir, 'receiver_2.pth.tar')).cuda()
test_dataset = DatasetFolder_eval(rx_signal_test_2, pilot_2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
model.eval()

t8 = time.time()
print("load model：{}s".format(t8 - t7))
output_all = []
with torch.no_grad():
    for idx, data in enumerate(test_loader):
        y, p = data
        y, p = y.cuda(), p.cuda()
        modelOutput = model(y, p)
        output_all.append(modelOutput)
output_all = torch.cat(output_all, dim=0)

t9 = time.time()
print("model eval：{}s".format(t9 - t8))
predict = torch.where(output_all >= 0, 1.0, 0.0)
score = torch.where(predict == tx_bits_test_2, 1.0, 0.0)
acc_2 = score.mean().cpu().numpy()
t10 = time.time()
print("count score：{}s".format(t10 - t9))
print('case2: acc= ' + str(acc_2))

# ======================================================================================================================
# Combined score
final_score = 0.4 * acc_1a + 0.1 * acc_1b + 0.5 * acc_2
print('The final score: acc =' + str(final_score))
t11 = time.time()
print('The total running time {}s'.format(t11 - t0))
