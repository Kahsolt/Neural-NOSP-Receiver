#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/26

from typing import *
from matplotlib.axes import Axes

from vis import *

# 若 Y = H * (sqrt(W) * QAM(bits) + sqrt(V) * P) + N
# 则 同样的bits + 同样的P 应该产生相同的Y？

dataset = 'D2'

with HDF5File(os.path.join('data', f'{dataset}.hdf5'), 'r') as fh:
  pilot     = np.asarray(fh['pilot'],     dtype=np.int8)    # [N=1, L=2/4, T=12, S=624/96, c=2], vset {-1, 0, 1}
  rx_signal = np.asarray(fh['rx_signal'], dtype=np.float16) # [N=20000, L=2/4, T=12, S=624/96, c=2]; float16
  tx_bits   = np.asarray(fh['tx_bits'],   dtype=np.int8)    # [N=20000, L=2/4, T=12, S=624/96, M=4/6]; vset {0, 1}, target label

# data cube: [N=20000, L=4, T=12, S=96, D=8]
X = np.concatenate([tx_bits, np.repeat(pilot, len(tx_bits), axis=0)], axis=-1)
N, L, T, S, D = X.shape
X_flat = X.reshape(N*L*T*S, D)
# find target (you can change this!)
target = np.asarray([0, 1, 0, 1, 0, 0, 1, 1], dtype=X.dtype)
# match
match = (X_flat == target).all(axis=-1)
print('ratio:', match.sum() / match.size)
match_cube = match.reshape(N, L, T, S)


if 'ablation on L':
  var_dim = L
  nrows = int(var_dim**0.5)
  ncols = int(np.ceil(var_dim / nrows))

  fig, axs = plt.subplots(nrows, ncols)
  for v in range(var_dim):
    vals = []
    for n, l, t, s in zip(*np.where(match_cube[:, v:v+1, :, :])):
      #val = rx_signal[n, l, t, s, 0]**2 + rx_signal[n, l, t, s, 1]**2
      val = np.angle(rx_signal[n, l, t, s, 0] + rx_signal[n, l, t, s, 1]*1j)
      vals.append(val)
    axs[v//ncols][v%ncols].hist(vals, bins=100)
  plt.show()
  plt.close()

if 'ablation on T':
  var_dim = T
  nrows = int(var_dim**0.5)
  ncols = int(np.ceil(var_dim / nrows))

  fig, axs = plt.subplots(nrows, ncols)
  for v in range(var_dim):
    vals = []
    for n, l, t, s in zip(*np.where(match_cube[:, :, v:v+1, :])):
      #val = rx_signal[n, l, t, s, 0]**2 + rx_signal[n, l, t, s, 1]**2
      val = np.angle(rx_signal[n, l, t, s, 0] + rx_signal[n, l, t, s, 1]*1j)
      vals.append(val)
    axs[v//ncols][v%ncols].hist(vals, bins=100)
  plt.show()
  plt.close()

if 'ablation on S':
  var_dim = S
  nrows = int(var_dim**0.5)
  ncols = int(np.ceil(var_dim / nrows))

  fig, axs = plt.subplots(nrows, ncols)
  for v in range(var_dim):
    vals = []
    for n, l, t, s in zip(*np.where(match_cube[:, :, :, v:v+1])):
      #val = rx_signal[n, l, t, s, 0]**2 + rx_signal[n, l, t, s, 1]**2
      val = np.angle(rx_signal[n, l, t, s, 0] + rx_signal[n, l, t, s, 1]*1j)
      vals.append(val)
    axs[v//ncols][v%ncols].hist(vals, bins=100)
  plt.show()
  plt.close()
