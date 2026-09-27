#!/bin/bash
# 大模型只当老师，提交的仍是 288x7（评测时间与 B 接近）。
# 先把本机更新后的 student.py / train.py / 两个 json / 本脚本传到云上，再执行。
set -e
cd "$(dirname "$0")"
python -c "import torch; assert torch.cuda.is_available()"
python -m unittest discover -s tests -v

echo "==== 老师：384x8，不提交，只用来蒸馏 ===="
python train.py --implementation student --config configs/teacher.json \
  --device cuda --seed 17 --steps 8000 --eval-every 1000 --ema 0.999 --lr 0.001 \
  --run-dir runs/teacher-8k

echo "==== 学生：与 B 同尺寸 + QK-Norm + 蒸馏 + SWA ===="
python train.py --implementation student --config configs/submit_distill.json \
  --device cuda --seed 17 --steps 12000 --eval-every 1000 --ema 0.999 --lr 0.001 \
  --teacher runs/teacher-8k/checkpoint.pt --distill 0.6 --distill-temp 2 --swa 6 \
  --run-dir runs/cloud-distill-12k

echo DONE
ls -lh runs/teacher-8k/checkpoint.pt runs/cloud-distill-12k/checkpoint.pt
echo "只下载 runs/cloud-distill-12k/checkpoint.pt 回本地做 CPU 测试。老师权重不用交。"
