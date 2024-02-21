#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/20

from modelTrain import *

BASE_PATH = Path(__name__).parent
LOG_PATH = BASE_PATH / 'lightning_logs'
SUBMIT_PATH = BASE_PATH / 'modelSubmit'

CASE_TO_LOG = {
  '1a': 'version_13',
  '1b': 'version_1?',    # 还没训
  '2':  'version_15',
}


for case, log_dn in CASE_TO_LOG.items():
  fp_dp = LOG_PATH / log_dn / 'checkpoints'
  if not fp_dp.exists(): continue
  fp_in = list(fp_dp.iterdir())[0]
  if not fp_in.exists(): continue

  config = CASE_TO_CONFIG[case]
  model_cls = globals()[config['model']]
  model = model_cls(**config)
  print(f'>> load from {fp_in}')
  lit: LitModel = LitModel.load_from_checkpoint(fp_in, model)
  model = lit.model.cpu()

  name = f'receiver_{case}'
  fp_out = SUBMIT_PATH / f'{name}.pth.tar'
  print(f'>> save to {fp_out}')
  torch.save(model, fp_out)

  fsize = os.path.getsize(fp_out) / 2**20
  try:
    if case == '1b':
      assert fsize <= 20
    else:
      assert fsize <= 100
  except AssertionError:
    print(f'>> fsize exceed limit: {fsize:.4f} MB')
