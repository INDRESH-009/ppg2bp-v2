# PPG → BP (SBP/DBP) CNIBP Model

- Subject-wise 5-fold CV
- Optional SQI gating (rejects low-quality windows)
- CNN + Transformer hybrid (strong on SBP)
- Weighted regression loss (SBP emphasized)

## Quickstart

```bash
# 0) create CV splits once
python make_splits.py --data_root ./data_root --out_dir ./splits --n_splits 5 --seed 42

# 1) train (5-fold)
python -m cnibp.train --data_root ./data_root --splits_dir ./splits --out_dir ./outputs \
  --use_sqi --sqi_thresh 0.8 --epochs 120 --batch_size 64 --lr 1e-4

# 2) evaluate (aggregates fold metrics)
python -m cnibp.evaluate --out_dir ./outputs

# 3) infer on one subject folder (or a single npz path)
python -m cnibp.infer --ckpt ./outputs/fold0/best.pt --input ./data_root/300001
