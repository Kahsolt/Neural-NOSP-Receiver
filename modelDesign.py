#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/20

import torch
import torch.nn as nn
from torch import Tensor


class ResBlock(nn.Module):

  def __init__(self, channel_list, H, W):
    super().__init__()

    self.channel_list = channel_list
    self.conv1 = nn.Conv2d(self.channel_list[2], self.channel_list[0], kernel_size=3, padding='same')
    self.ln1 = nn.LayerNorm([self.channel_list[0], H, W])
    self.conv2 = nn.Conv2d(self.channel_list[0], self.channel_list[1], kernel_size=3, padding='same')
    self.ln2 = nn.LayerNorm([self.channel_list[1], H, W])
    self.act = nn.ReLU()

  def forward(self, x:Tensor) -> Tensor:
    r = x
    x = self.ln1(x)
    x = self.act(x)
    x = self.conv1(x)
    x = self.ln2(x)
    x = self.act(x)
    x = self.conv2(x)
    o = r + x
    return o


class Neural_receiver(nn.Module):

  def __init__(self, subcarriers:int, timesymbols:int, streams:int, num_bits_per_symbol:int, **kwargs):
    super().__init__()

    self.subcarriers = subcarriers                  # S=624/96
    self.timesymbols = timesymbols                  # T=12
    self.streams = streams                          # L=Nr, 2/4
    self.num_bits_per_symbol = num_bits_per_symbol  # M=4/6

    self.num_blocks = 6
    self.channel_list = [24, 24, 24]

    self.pre_conv = nn.Conv2d(4 * self.streams, self.channel_list[2], kernel_size=3, padding='same')
    self.blocks = nn.ModuleList([
      ResBlock(self.channel_list, H=self.timesymbols, W=self.subcarriers) for _ in range(self.num_blocks)
    ])
    self.post_conv = nn.Conv2d(self.channel_list[1], self.streams * self.num_bits_per_symbol, kernel_size=3, padding='same')

  def forward(self, x:Tensor, t:Tensor) -> Tensor:
    # [B=16, L=2, T=12, S=642, c=2]
    # x: vrng ~ ±7.0
    # t: vrng {-1, 1}

    if not 'use jit model':
      B, L, T, S, c = x.shape
      M = self.num_bits_per_symbol
      assert L == self.streams
      assert T == self.timesymbols
      assert S == self.subcarriers
    else:
      B = x.shape[0]
      M = self.num_bits_per_symbol
      L = self.streams
      T = self.timesymbols
      S = self.subcarriers
      c = 2

    # [B, L*c, T, S]
    x = x.permute(0, 1, 4, 2, 3).reshape(B, L * c, T, S)
    t = t.permute(0, 1, 4, 2, 3).reshape(B, L * c, T, S)
    # [B, L*4, T, S]
    z = torch.cat([x, t], dim=1)

    # [B, C=L*4, T, S]; T for time-domain, S for freq-domain
    z = self.pre_conv(z)      # [B, C=24, T=12, F=624/96]
    for block in self.blocks:
      z = block(z)            # [B, C=24, T, S]
    z = self.post_conv(z)     # [B, C=L*M, T, S]

    # [B, L, T, S, M]
    z: Tensor
    z = z.reshape(B, L, M, T, S).permute(0, 1, 3, 4, 2)

    return z


if __name__ == '__main__':
  model = Neural_receiver(642, 12, 2, 4)
  X = torch.rand([16, 2, 12, 642, 2])
  T = (torch.rand([16, 2, 12, 642, 2]) > 0) * 2 - 1
  logits = model(X, T)
  print('X.shape:', X.shape)
  print('logits.shape:', logits.shape)

  model = Neural_receiver(96, 12, 4, 6)
  X = torch.rand([16, 4, 12, 96, 2])
  T = (torch.rand([16, 4, 12, 96, 2]) > 0) * 2 - 1
  logits = model(X, T)
  print('X.shape:', X.shape)
  print('logits.shape:', logits.shape)
