#!/bin/bash
# 第三轮：更大老师 + 不放回窗口。学生仍用上一轮最好的 288x7 / dropout 0.15。
# 必须重传更新后的 train.py。
set -e
cd "$(dirname "$0")"
python -c "import torch; assert torch.cuda.is_available()"
python -m unittest discover -s tests -v

echo "==== 老师：512x8，12000 步（不提交） ===="
python train.py --implementation student --config configs/teacher_large.json \
  --device cuda --seed 17 --steps 12000 --eval-every 1000 --ema 0.999 --lr 0.0008 \
  --unique-windows --run-dir runs/teacher-512-12k

python -c "import json; m=json.load(open('runs/teacher-512-12k/metrics.json')); print('TEACHER_VAL', m['validation'].get('selected_bpb', m['validation']['bpb']))"

echo "==== 学生：沿用 submit_distill.json，蒸馏 14000 步 ===="
python train.py --implementation student --config configs/submit_distill.json \
  --device cuda --seed 17 --steps 14000 --eval-every 1000 --ema 0.999 --lr 0.001 \
  --teacher runs/teacher-512-12k/checkpoint.pt --distill 0.7 --distill-temp 2 --swa 6 \
  --unique-windows --run-dir runs/cloud-distill-14k

echo DONE
ls -lh runs/teacher-512-12k/checkpoint.pt runs/cloud-distill-14k/checkpoint.pt
echo "只下载 runs/cloud-distill-14k/checkpoint.pt。请同时看上面的 TEACHER_VAL。"
