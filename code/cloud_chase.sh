#!/bin/bash
# 冲 1.4：短日程 + EMA。必须先把本机更新后的 train.py 传到云上再跑。
set -e
cd "$(dirname "$0")"
python -c "import torch; assert torch.cuda.is_available()"
python -m unittest discover -s tests -v

echo "==== A: 已验证的 256 宽模型 + EMA + 15000 步 ===="
python train.py --implementation student --config configs/submit_dropout.json \
  --device cuda --seed 17 --steps 15000 --eval-every 1000 --ema 0.999 \
  --run-dir runs/cloud-ema-15k

echo "==== B: 更大模型短训 + 更强 dropout + EMA + 10000 步 ===="
python train.py --implementation student --config configs/submit_chase.json \
  --device cuda --seed 17 --steps 10000 --eval-every 1000 --ema 0.999 --lr 0.0012 \
  --run-dir runs/cloud-chase-10k

echo DONE
ls -lh runs/cloud-ema-15k/checkpoint.pt runs/cloud-chase-10k/checkpoint.pt
