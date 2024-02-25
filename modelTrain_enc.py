#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/20

# Y = H * (sqrt(W) * QAM(bits) + sqrt(V) * P) + N
# try train this H

from modelTrain import *
from torchmetrics.regression import MeanSquaredError
from QAM16 import QAM_Modulation


CASE_TO_CONFIG = {
  '1a': {
    'model': 'NeuralReceiver_1a',
    'subcarriers': 624,         # 子载波 S
    'timesymbols': 12,          # 符号 T
    'streams': 2,               # 接收天线/传输层 L
    'num_bits_per_symbol': 4,   # 比特 M
  },
  '1b': {
    'model': 'NeuralReceiver_1b',
    'subcarriers': 624,
    'timesymbols': 12,
    'streams': 2,
    'num_bits_per_symbol': 4,
  },
  '2': {
    'model': 'NeuralReceiver_1a',   # NeuralReceiver_2
    'subcarriers': 96,
    'timesymbols': 12,
    'streams': 4,
    'num_bits_per_symbol': 6,
  },
}
CASE_TO_DATASET = {
  '1a': 'D1',
  '1b': 'D1',
  '2':  'D2',
}


class QAMSignalDataset(Dataset):

  def __init__(self, rx_signal:ndarray, pilot:ndarray, tx_bits:ndarray, M:int) -> None:
    super().__init__()

    self.X = torch.from_numpy(QAM_Modulation(tx_bits, M, 0.1)).float()
    self.P = torch.from_numpy(pilot).float()
    self.Y = torch.from_numpy(rx_signal).float()

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx) -> Tuple[Tensor, Tensor, Tensor]:
    x = self.X[idx]
    y = self.Y[idx]
    return x, self.P, y


class MLP_INI(nn.Module):

  def __init__(self, d_in:int):
    super().__init__()

    self.conv1 = nn.Conv2d(d_in, d_in, kernel_size=1, padding='same')
    self.act = nn.LeakyReLU()
    self.conv2 = nn.Conv2d(d_in, d_in, kernel_size=1, padding='same')

  def forward(self, x:Tensor) -> Tensor:
    x = self.conv1(x)
    x = self.act(x)
    x = self.conv2(x)
    return x


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


class ModelEnc(nn.Module):

  def __init__(self, subcarriers:int, timesymbols:int, streams:int, num_bits_per_symbol:int, **kwargs):
    super().__init__()

    self.subcarriers = subcarriers                  # S=624/96
    self.timesymbols = timesymbols                  # T=12
    self.streams = streams                          # L=Nr, 2/4
    self.num_bits_per_symbol = num_bits_per_symbol  # M=4/6

    self.sqrtW = nn.Parameter(torch.FloatTensor([0.9]).sqrt_(), requires_grad=False)
    self.sqrtV = nn.Parameter(torch.FloatTensor([0.1]).sqrt_(), requires_grad=False)

    self.num_blocks = 6
    self.channel_list = [32, 32, 32]

    self.pre_mlp = MLP_INI(2 * self.streams)
    self.pre_conv = nn.Conv2d(2 * self.streams, self.channel_list[2], kernel_size=3, padding='same')
    self.blocks = nn.ModuleList([
      ResBlock(self.channel_list, H=self.timesymbols, W=self.subcarriers) for _ in range(self.num_blocks)
    ])
    self.post_conv = nn.Conv2d(self.channel_list[1], self.streams * 2, kernel_size=3, padding='same')

  def forward(self, x:Tensor, t:Tensor) -> Tensor:
    # [B=16, L=2, T=12, S=642, c=2]

    B = x.shape[0]
    L = self.streams
    T = self.timesymbols
    S = self.subcarriers
    c = 2

    # [B, L*c, T, S]
    x = x.permute(0, 1, 4, 2, 3).reshape(B, L * c, T, S)
    P = t.permute(0, 1, 4, 2, 3).reshape(B, L * c, T, S)

    # [B, L*c, T, S]
    D = self.pre_mlp(x)
    z = self.sqrtW * D + self.sqrtV * P

    # [B, C=L*c, T, S]; T for time-domain, S for freq-domain
    z = self.pre_conv(z)      # [B, C=24, T=12, F=624/96]
    for block in self.blocks:
      z = block(z)            # [B, C=24, T, S]
    z = self.post_conv(z)     # [B, C=L*2, T, S]

    # [B, L, T, S, c]
    z: Tensor
    z = z.reshape(B, L, c, T, S).permute(0, 1, 3, 4, 2)

    return z


class LitModel(LightningModule):

  def __init__(self, model:NeuralReceiverBase):
    super().__init__()

    self.model = model

    # ↓↓ training specified ↓↓
    self.epochs = -1
    self.lr = 2e-4
    self.train_mse = MeanSquaredError()
    self.valid_mse = MeanSquaredError()

  def setup_train_args(self, args):
    self.epochs = args.epochs
    self.lr = args.lr

  def configure_optimizers(self) -> Optimizer:
    optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs, verbose=True)
    return {
      'optimizer': optimizer,
      'lr_scheduler': scheduler,
    }

  def optimizer_step(self, epoch:int, batch_idx:int, optim:Optimizer, optim_closure:Callable):
    super().optimizer_step(epoch, batch_idx, optim, optim_closure)
    if batch_idx % 10 == 0:
      self.log_dict({f'lr-{i}': group['lr'] for i, group in enumerate(optim.param_groups)})

  def training_step(self, batch:Tuple[Tensor], batch_idx:int) -> Tensor:
    x, t, y = batch
    outputs = self.model(x, t)
    loss = F.mse_loss(outputs, y)

    self.log('train/loss', loss, on_step=True, on_epoch=True)
    self.train_mse(outputs.flatten(1), y.flatten(1))
    self.log('train/mse', self.train_mse, on_step=True, on_epoch=True)
    return loss

  def validation_step(self, batch:Tuple[Tensor], batch_idx:int) -> Tensor:
    x, t, y = batch
    outputs = self.model(x, t)
    loss = F.mse_loss(outputs, y)

    self.log('valid/loss', loss, on_step=True, on_epoch=True)
    self.valid_mse(outputs.flatten(1), y.flatten(1))
    self.log('valid/mse', self.train_mse, on_step=True, on_epoch=True)
    return loss



def train(args):
  seed_everything(args.seed)
  print('>> cmd:', ' '.join(sys.argv))
  print('>> args:', vars(args))

  ''' Data '''
  dataset = CASE_TO_DATASET[args.case]
  with HDF5File(os.path.join('data', f'{dataset}.hdf5'), 'r') as fh:
    pilot     = np.asarray(fh['pilot']).squeeze(axis=0)  # [L=2/4, T=12, S=624/96, c=2], vset {-1, 0, +1}
    rx_signal = np.asarray(fh['rx_signal'])     # [N=20000, L=2/4, T=12, S=624/96, c=2]; float16
    tx_bits   = np.asarray(fh['tx_bits'])       # [N=20000, L=2/4, T=12, S=624/96, M=4/6]; vset {0, 1}, target label
  print('rx_signal:', rx_signal.shape, rx_signal.dtype)
  print('tx_bits:', tx_bits.shape, tx_bits.dtype)
  print('pilot:', pilot.shape, pilot.dtype)
  n_samples = rx_signal.shape[0]

  cp = int(n_samples * args.split_ratio)
  rx_signal_train = rx_signal[:cp]
  rx_signal_valid = rx_signal[cp:]
  tx_bits_train = tx_bits[:cp]
  tx_bits_valid = tx_bits[cp:]

  M = CASE_TO_CONFIG[args.case]['num_bits_per_symbol']
  trainset = QAMSignalDataset(rx_signal_train, pilot, tx_bits_train, M)
  validset = QAMSignalDataset(rx_signal_valid, pilot, tx_bits_valid, M)
  dataloader_kwargs = {
    'num_workers': 0,
    'persistent_workers': False,
    'pin_memory': True,
  }
  trainloader = DataLoader(trainset, args.batch_size, shuffle=True,  drop_last=True,  **dataloader_kwargs)
  validloader = DataLoader(validset, args.batch_size, shuffle=False, drop_last=False, **dataloader_kwargs)

  ''' Model & Optim '''
  config = CASE_TO_CONFIG[args.case]
  #model_cls = globals()[config['model']]
  model_cls = ModelEnc
  model = model_cls(**config)
  print(model)
  lit = LitModel(model)
  if args.load:
    lit = LitModel.load_from_checkpoint(args.load, model=model)
  lit.setup_train_args(args)

  ''' Train '''
  trainer = Trainer(
    max_epochs=args.epochs,
    precision='16-mixed',
    benchmark=True,
    enable_checkpointing=True,
  )
  trainer.fit(lit, trainloader, validloader)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-C', '--case',       default='2', choices=['1a', '1b', '2'])
  parser.add_argument('-B', '--batch_size', default=128,  type=int)
  parser.add_argument('-E', '--epochs',     default=30,   type=int)
  parser.add_argument('-lr', '--lr',        default=1e-3, type=eval)
  parser.add_argument('--split_ratio',      default=0.8,  type=float)
  parser.add_argument('--load', type=Path)
  parser.add_argument('--seed', type=int, default=114514)
  args = parser.parse_args()

  train(args)
