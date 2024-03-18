#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/20

from modelTrain import *


class SignalDataset(Dataset):

  def __init__(self, rx_signal:ndarray, pilot:ndarray, tx_bits:ndarray) -> None:
    super().__init__()

    self.x = torch.from_numpy(rx_signal).float()
    self.t = torch.from_numpy(pilot).float()
    bits = torch.from_numpy(tx_bits).long()   # [N, L, T, S, M]
    self.bits = bits
    y = 0
    for i in range(bits.shape[-1]):
      y += bits[..., i] * 2**i
    self.y = y    # [N, L, T, S]

  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx) -> Tuple[Tensor, Tensor, Tensor]:
    x = self.x[idx]
    y = self.y[idx]
    bits = self.bits[idx]
    return x, self.t, y, bits


class LitModel(LightningModule):

  def __init__(self, model:NeuralReceiverBase, args=None):
    super().__init__()

    self.model = model
    if args: self.save_hyperparameters(args)

    # ↓↓ training specified ↓↓
    self.epochs = -1
    self.lr = 2e-4
    self.train_acc = BinaryAccuracy()
    self.valid_acc = BinaryAccuracy()

  def setup_train_args(self, args):
    self.epochs = args.epochs
    self.lr = args.lr

  def configure_optimizers(self) -> Optimizer:
    return Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)

  def training_step(self, batch:Tuple[Tensor], batch_idx:int) -> Tensor:
    x, t, y, bits = batch
    logits = self.model(x, t)
    loss = F.cross_entropy(logits, y)
    self.log('train/loss', loss, on_step=True, on_epoch=True)

    preds = logits.argmax(dim=1)
    pred_bits = []
    for _ in range(bits.shape[-1]):
      pred_bits.append(preds % 2)
      preds = preds // 2
    pred_bits = torch.stack(list(reversed(pred_bits)), dim=-1)
    self.train_acc(pred_bits, bits)
    self.log('train/acc', self.train_acc, on_step=True, on_epoch=True)
    return loss

  def validation_step(self, batch:Tuple[Tensor], batch_idx:int) -> Tensor:
    x, t, y, bits = batch
    logits = self.model(x, t)
    loss = F.cross_entropy(logits, y)
    self.log('valid/loss', loss, on_step=True, on_epoch=True)

    preds = logits.argmax(dim=1)
    pred_bits = []
    for _ in range(bits.shape[-1]):
      pred_bits.append(preds % 2)
      preds = preds // 2
    pred_bits = torch.stack(list(reversed(pred_bits)), dim=-1)
    self.valid_acc(pred_bits, bits)
    self.log('valid/acc', self.valid_acc, on_step=True, on_epoch=True)
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

  trainset = SignalDataset(rx_signal_train, pilot, tx_bits_train)
  validset = SignalDataset(rx_signal_valid, pilot, tx_bits_valid)
  dataloader_kwargs = {
    'num_workers': 0,
    'persistent_workers': False,
    'pin_memory': True,
  }
  trainloader = DataLoader(trainset, args.batch_size, shuffle=True,  drop_last=True,  **dataloader_kwargs)
  validloader = DataLoader(validset, args.batch_size, shuffle=False, drop_last=False, **dataloader_kwargs)

  ''' Model & Optim '''
  config = CASE_TO_CONFIG[args.case]
  model_cls = globals()[args.model or config['model']]
  model = model_cls(**config)
  print(model)
  if args.load:
    lit = LitModel.load_from_checkpoint(args.load, model=model, args=args)
  else:
    lit = LitModel(model, args)
  lit.setup_train_args(args)

  ''' Train '''
  save_ckpt_callback = ModelCheckpoint(monitor='valid/acc', mode='max')
  trainer = Trainer(
    max_epochs=args.epochs,
    precision='16-mixed',
    benchmark=True,
    callbacks=[save_ckpt_callback],
    log_every_n_steps=10,
  )
  trainer.fit(lit, trainloader, validloader)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M', '--model', default='Neural_receiver_clf16', help='model arch to overwrite --case')
  parser.add_argument('-C', '--case',       default='2', choices=['1a', '1b', '2'])
  parser.add_argument('-B', '--batch_size', default=16,   type=int)
  parser.add_argument('-E', '--epochs',     default=30,   type=int)
  parser.add_argument('-lr', '--lr',        default=1e-3, type=eval)
  parser.add_argument('--split_ratio',      default=0.99, type=float)
  parser.add_argument('--load', type=Path)
  parser.add_argument('--seed', type=int, default=114514)
  args = parser.parse_args()

  train(args)
