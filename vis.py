#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/21

import os
from argparse import ArgumentParser

import numpy as np
from numpy import ndarray
from h5py import File as HDF5File
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def to_TSC(x:ndarray) -> ndarray:
  L, T, S, c = x.shape
  return x.transpose(1, 2, 3, 0).reshape(T, S, L*c)


def vis_pilot(pilot:ndarray):
  vmin, vmax = pilot.min(), pilot.max()
  L, T, S, C = pilot.shape
  plt.clf()
  fig, axes = plt.subplots(L, C)
  for l in range(L):
    for c in range(C):
      ax: Axes = axes[l, c]
      sns.heatmap(pilot[l, :, :, c], vmin=vmin, vmax=vmax, cmap='coolwarm', cbar=False, ax=ax)
      ax.set_title(f'layer-{l} {"real" if c == 0 else "imag"}')
  plt.suptitle('pilot')
  plt.tight_layout()
  plt.show()


def vis_tx(tx_bits:ndarray):
  for i, tx in enumerate(tx_bits):
    L, T, S, M = tx.shape
    plt.clf()
    fig, axes = plt.subplots(L * M)
    for l in range(L):
      for m in range(M):
        ax: Axes = axes[l * M + m]
        ax.imshow(tx[l, :, :, m], cmap='grey')
        ax.set_axis_off()
        ax.set_title(f'layer-{l} bit-{m}')
    plt.suptitle(f'[{i}] tx - bits')
    plt.tight_layout()
    plt.show()


def vis_rx(rx_signal:ndarray):
  for i, rx in enumerate(rx_signal):
    L, T, S, C = rx.shape
    vmin, vmax = rx.min(), rx.max()
    plt.clf()
    fig, axes = plt.subplots(L, C)
    for l in range(L):
      for c in range(C):
        ax: Axes = axes[l, c]
        sns.heatmap(rx[l, :, :, c], vmin=vmin, vmax=vmax, cbar=False, ax=ax)
        ax.set_axis_off()
        ax.set_title(f'layer-{l} {"real" if c == 0 else "imag"}')
    plt.suptitle(f'[{i}] rx - coords')
    plt.tight_layout()
    plt.show()


def vis_rx_hist(rx_signal:ndarray):
  for rx in rx_signal:
    L, T, S, C = rx.shape
    plt.clf()
    fig, axes = plt.subplots(L, C)
    for l in range(L):
      for c in range(C):
        ax: Axes = axes[l, c]
        ax.hist(rx[l, :, :, c].flatten(), bins=100)
        ax.set_title(f'layer-{l} {"real" if c == 0 else "imag"}')
    plt.suptitle('rx - coords hist')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-D', '--dataset', default='D2', choices=['D1', 'D2'])
  parser.add_argument('-P', '--pilot', action='store_true')
  parser.add_argument('-T', '--tx', action='store_true')
  parser.add_argument('-R', '--rx', action='store_true')
  parser.add_argument('-RH', '--rx_hist', action='store_true')
  args = parser.parse_args()

  assert any([args.pilot, args.tx, args.rx, args.rx_hist]), 'at least specify one --pilot, --tx, --rx, --rx_hist'

  with HDF5File(os.path.join('data', f'{args.dataset}.hdf5'), 'r') as fh:
    pilot     = np.asarray(fh['pilot']).squeeze(axis=0)  # [L=2/4, T=12, S=624/96, c=2], vset {-1, 0, 1}
    rx_signal = np.asarray(fh['rx_signal'])     # [N=20000, L=2/4, T=12, S=624/96, c=2]; float16
    tx_bits   = np.asarray(fh['tx_bits'])       # [N=20000, L=2/4, T=12, S=624/96, M=4/6]; vset {0, 1}, target label

  if args.pilot: vis_pilot(pilot)
  if args.tx: vis_tx(tx_bits)
  if args.rx: vis_rx(rx_signal)
  if args.rx_hist: vis_rx_hist(rx_signal)
