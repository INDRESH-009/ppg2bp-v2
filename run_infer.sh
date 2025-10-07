#!/usr/bin/env bash
python -m cnibp.infer \
  --ckpt ./outputs/fold0/best.pt \
  --input ./data_root/300001
