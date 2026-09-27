#!/bin/bash
# 第二轮蒸馏：老师多训一会，学生加长到 2 万步。提交尺寸仍是 288x7。
# 云上已有 teacher.json / student.py / train.py 即可；只需再传本脚本和 submit_distill2.json。
set -e
cd "$(dirname "$0")"
python -c "import torch; assert torch.cuda.is_available()"
python -m unittest discover -s tests -v

echo "==== 老师：384x8，15000 步（仍不提交） ===="
python train.py --implementation student --config configs/teacher.json \
  --device cuda --seed 17 --steps 15000 --eval-every 1000 --ema 0.999 --lr 0.001 \
  --run-dir runs/teacher-15k

echo "==== 学生：同尺寸 + 更强 dropout + 蒸馏 20000 步 + SWA ===="
python train.py --implementation student --config configs/submit_distill2.json \
  --device cuda --seed 17 --steps 20000 --eval-every 1000 --ema 0.999 --lr 0.0009 \
  --teacher runs/teacher-15k/checkpoint.pt --distill 0.65 --distill-temp 2 --swa 8 \
  --run-dir runs/cloud-distill-20k

echo DONE
ls -lh runs/teacher-15k/checkpoint.pt runs/cloud-distill-20k/checkpoint.pt
echo "只下载 runs/cloud-distill-20k/checkpoint.pt。老师不用交。"
