#!/bin/bash
# 冲分版：更大模型 + 5 万步。在云上 code 目录执行：bash cloud_push.sh
set -e
cd "$(dirname "$0")"

python -c "import torch; assert torch.cuda.is_available(), '没有 GPU'"
python -m unittest discover -s tests -v

python train.py \
  --implementation student \
  --config configs/submit_push.json \
  --device cuda \
  --seed 17 \
  --steps 50000 \
  --eval-every 2000 \
  --run-dir runs/cloud-push-50k

echo "下载 runs/cloud-push-50k/checkpoint.pt 后必须在自己电脑 CPU 上测测试集。"
echo "若测试评测超过约 68 秒，不要提交这一版。"
ls -lh runs/cloud-push-50k/checkpoint.pt
