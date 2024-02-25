import numpy as np
from numpy import ndarray
from h5py import File as HDF5File


def QAM_Modulation(tx_bits:ndarray, M:int, A:float) -> ndarray:
  tx_symbols = np.empty([*tx_bits.shape[:-1], 2], dtype=np.float16)
  I = tx_bits[..., 0::2]   # 按照 3GPP 标准间隔取下标 0 和 2
  Q = tx_bits[..., 1::2]   # 按照 3GPP 标准间隔取下标 1 和 3
  tx_symbols[..., 0] = ((-1) ** (I[..., 0])) * ((2 * (I[..., 1] - 1)) - 1 + M)  # I路 
  tx_symbols[..., 1] = ((-1) ** (Q[..., 0])) * ((2 * (Q[..., 1] - 1)) - 1 + M)  # Q路
  return A * tx_symbols


if __name__ == '__main__':
  with HDF5File('./data/D1.hdf5', 'r') as fh:
    tx_bits = np.asarray(fh["tx_bits"])
  print('tx_bits.shape:', tx_bits.shape)

  tx_symbols = QAM_Modulation(tx_bits, 4, 0.1)
  print('tx_symbols.shape:', tx_symbols.shape)
  print(tx_symbols)
