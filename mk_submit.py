#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/20

from modelTrain import *

BASE_PATH = Path(__name__).parent
LOG_PATH = BASE_PATH / 'lightning_logs'
SUBMIT_PATH = BASE_PATH / 'modelSubmit'

CASE_TO_LOG = {
  'receiver_1a': 'version_13',
  'receiver_1b': 'version_1?',    # 还没训
  'receiver_2':  'version_15',
}
CASE_TO_DATASET = {
  'receiver_1a': 'D1',
  'receiver_1b': 'D1',
  'receiver_2':  'D2',
}

for name, log_dn in CASE_TO_LOG.items():
  fp_dp = LOG_PATH / log_dn / 'checkpoints'
  if not fp_dp.exists(): continue
  fp_in = list(fp_dp.iterdir())[0]
  if not fp_in.exists(): continue

  dataset = CASE_TO_DATASET[name]
  schema = DATASET_TO_MODEL_SCHEMA[dataset]
  model = Neural_receiver(**schema)
  print(f'>> load from {fp_in}')
  lit: LitModel = LitModel.load_from_checkpoint(fp_in, model)
  model = lit.model.cpu()

  fp_out = SUBMIT_PATH / f'{name}.pth.tar'
  print(f'>> save to {fp_out}')
  torch.save(model, fp_out)

  fsize = os.path.getsize(fp_out) / 2**20
  try:
    if name == 'receiver_1b':
      assert fsize <= 20
    else:
      assert fsize <= 100
  except AssertionError:
    print(f'>> fsize exceed limit: {fsize:.4f} MB')
