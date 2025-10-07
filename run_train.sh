
---

# 2) `run_train.sh` (optional helper)

```bash
#!/usr/bin/env bash
python -m cnibp.train \
  --data_root ./data_root \
  --splits_dir ./splits \
  --out_dir ./outputs \
  --use_sqi --sqi_thresh 0.8 \
  --epochs 120 --batch_size 64 --lr 1e-4 --num_workers 6
