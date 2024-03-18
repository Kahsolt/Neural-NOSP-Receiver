#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/20

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils import weight_norm


''' baseline '''

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


class Neural_receiver_PE(nn.Module):

  def __init__(self, subcarriers:int, timesymbols:int, streams:int, num_bits_per_symbol:int, **kwargs):
    super().__init__()

    self.subcarriers = subcarriers                  # S=624/96
    self.timesymbols = timesymbols                  # T=12
    self.streams = streams                          # L=Nr, 2/4
    self.num_bits_per_symbol = num_bits_per_symbol  # M=4/6

    self.num_blocks = 6
    self.channel_list = [128, 128, 128]

    d_posenc = 32
    self.posenc = nn.Parameter(torch.empty([d_posenc, timesymbols, subcarriers]).normal_(std=0.02), requires_grad=True)
    self.pre_conv = nn.Conv2d(4 * self.streams+d_posenc, 128, kernel_size=self.channel_list[2], padding='same')
    self.blocks = nn.ModuleList([
      ResBlock(self.channel_list, H=self.timesymbols, W=self.subcarriers) for _ in range(self.num_blocks)
    ])
    self.post_conv = nn.Conv2d(self.channel_list[1], self.streams * self.num_bits_per_symbol, kernel_size=7, padding='same')

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
    # [B, d, T, S]
    posenc_ex = self.posenc.unsqueeze(0).expand(B, -1, -1, -1)
    # [B, L*4+d, T, S]
    z = torch.cat([z, posenc_ex], dim=1)

    # [B, C=L*4+d, T, S]; T for time-domain, S for freq-domain
    z = self.pre_conv(z)      # [B, C=24, T=12, F=624/96]
    for block in self.blocks:
      z = block(z)            # [B, C=24, T, S]
    z = self.post_conv(z)     # [B, C=L*M, T, S]

    # [B, L, T, S, M]
    z: Tensor
    z = z.reshape(B, L, M, T, S).permute(0, 1, 3, 4, 2)

    return z


''' ours '''

def init_weights(m:nn.Conv2d, mean:float=0.0, std:float=0.01):
  classname = m.__class__.__name__
  if 'Conv' in classname:
    m.weight.data.normal_(mean, std)

def get_padding(kernel_size:int, dilation:int=1):
  return int((kernel_size * dilation - dilation) / 2)


class ResConv(nn.Module):

  def __init__(self, d_in:int=128, k:int=3):
    super().__init__()

    self.convs = nn.ModuleList([
      weight_norm(nn.Conv2d(d_in, d_in, kernel_size=k, dilation=1, padding='same')),
      weight_norm(nn.Conv2d(d_in, d_in, kernel_size=k, dilation=2, padding=get_padding(k, 2))),
    ])
    self.convs.apply(init_weights)
    self.act = nn.LeakyReLU()

  def forward(self, x:Tensor) -> Tensor:
    for c in self.convs:
      xt = self.act(x)
      xt = c(xt)
      x = xt + x
    return x


class NeuralReceiverBase(nn.Module):

  def __init__(self, subcarriers:int, timesymbols:int, streams:int, num_bits_per_symbol:int):
    super().__init__()

    self.subcarriers = subcarriers                  # S=624/96
    self.timesymbols = timesymbols                  # T=12
    self.streams = streams                          # L=Nr, 2/4
    self.num_bits_per_symbol = num_bits_per_symbol  # M=4/6

  def forward(self, x:Tensor, t:Tensor) -> Tensor:
    raise NotImplementedError


class NeuralReceiver_1a(NeuralReceiverBase):
  
  def __init__(self, subcarriers:int, timesymbols:int, streams:int, num_bits_per_symbol:int, **kwargs):
    super().__init__(subcarriers, timesymbols, streams, num_bits_per_symbol)

    self.invH_ch = 128
    self.invQAM_ch = 128
    self.sqrtW = nn.Parameter(torch.FloatTensor([0.9]).sqrt_(), requires_grad=False)
    self.sqrtV = nn.Parameter(torch.FloatTensor([0.1]).sqrt_(), requires_grad=False)

    # [B, L*c, T, S] -> [B, L*c, T, S]
    self.invH = nn.Sequential(
      # pre_conv
      weight_norm(nn.Conv2d(self.streams*2, self.invH_ch, kernel_size=3, padding='same')),
      # resblocks
      ResConv(self.invH_ch, k=3),
      ResConv(self.invH_ch, k=3),
      # post_conv
      nn.LeakyReLU(),
      weight_norm(nn.Conv2d(self.invH_ch, self.streams*2, kernel_size=3, padding='same')),
    )
    # [B, L*c, T, S] -> [B, L*M, T, S]
    self.invQAM = nn.Sequential(
      # pre_conv
      weight_norm(nn.Conv2d(self.streams*2, self.invQAM_ch, kernel_size=3, padding='same')),
      # resblocks
      ResConv(self.invQAM_ch, k=3),
      ResConv(self.invQAM_ch, k=3),
      ResConv(self.invQAM_ch, k=3),
      ResConv(self.invQAM_ch, k=3),
      # post_conv
      nn.LeakyReLU(),
      weight_norm(nn.Conv2d(self.invQAM_ch, self.streams*self.num_bits_per_symbol, kernel_size=3, padding='same')),
    )

    self.invH[0].apply(init_weights)
    self.invH[-1].apply(init_weights)
    self.invQAM[0].apply(init_weights)
    self.invQAM[-1].apply(init_weights)

  def forward(self, x:Tensor, t:Tensor) -> Tensor:
    # [B=16, L=2, T=12, S=642, c=2]
    # x: vrng ~ ±7.0
    # t: vrng {-1, 1}

    if not 'dynamic':
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
    Y = x.permute(0, 1, 4, 2, 3).reshape(B, L * c, T, S)
    P = t.permute(0, 1, 4, 2, 3).reshape(B, L * c, T, S)

    # bits = invQAM(((Y - N) * invH - sqrt(V) * P) / sqrt(W))
    z = self.invQAM((self.invH(Y) - self.sqrtV * P) / self.sqrtW)

    # [B, L, T, S, M]
    z: Tensor
    z = z.reshape(B, L, M, T, S).permute(0, 1, 3, 4, 2)

    return z


class NeuralReceiver_1b(NeuralReceiverBase):

  def __init__(self, subcarriers:int, timesymbols:int, streams:int, num_bits_per_symbol:int, **kwargs):
    super().__init__(subcarriers, timesymbols, streams, num_bits_per_symbol)

  def forward(self, x:Tensor, t:Tensor) -> Tensor:
    M = self.num_bits_per_symbol
    return x[:, :, :, :, :1].expand(-1, -1, -1, -1, M)


class NeuralReceiver_2(NeuralReceiverBase):

  def __init__(self, subcarriers:int, timesymbols:int, streams:int, num_bits_per_symbol:int, **kwargs):
    super().__init__(subcarriers, timesymbols, streams, num_bits_per_symbol)

  def forward(self, x:Tensor, t:Tensor) -> Tensor:
    M = self.num_bits_per_symbol
    return x[:, :, :, :, :1].expand(-1, -1, -1, -1, M)


if __name__ == '__main__':
  from modelTrain import CASE_TO_CONFIG

  def config_to_input_shape(config):
    S = config['subcarriers']
    T = config['timesymbols']
    L = config['streams']
    M = config['num_bits_per_symbol']
    return L, T, S, M

  # baseline
  print('===== [baseline] =====')
  for case in ['1a', '1b', '2']:
    print(f'[Case {case}]')
    config = CASE_TO_CONFIG[case]
    model = Neural_receiver(**config)
    input_shape = config_to_input_shape(config)
    X = torch.randn([4, *input_shape])
    T = (X > 0) * 2 - 1
    logits = model(X, T)
    print('X.shape:', X.shape)
    print('logits.shape:', logits.shape)
    assert logits.shape[-1] == config['num_bits_per_symbol']

  # ours
  print('===== [ours] =====')
  for case in ['1a', '1b', '2']:
    print(f'[Case {case}]')
    config = CASE_TO_CONFIG[case]
    model_cls = globals()[config['model']]
    model = model_cls(**config)
    input_shape = config_to_input_shape(config)
    X = torch.randn([4, *input_shape])
    T = (X > 0) * 2 - 1
    logits = model(X, T)
    print('X.shape:', X.shape)
    print('logits.shape:', logits.shape)
    assert logits.shape[-1] == config['num_bits_per_symbol']
