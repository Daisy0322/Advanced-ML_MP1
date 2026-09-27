#!/bin/bash
# 在 AutoDL 终端里：进入本脚本所在的 code 目录后执行  bash cloud_run.sh
set -e
cd "$(dirname "$0")"

echo "==== GPU ===="
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo "==== 安装 PyTorch 2.7.1 CUDA ===="
python -m pip install -q torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -q -r requirements.txt

echo "==== 合同测试 ===="
python -m unittest discover -s tests -v

echo "==== 开始训练 20000 步 ===="
python train.py \
  --implementation student \
  --config configs/submit_dropout.json \
  --device cuda \
  --seed 17 \
  --steps 20000 \
  --eval-every 1000 \
  --run-dir runs/cloud-20k

echo "==== 完成 ===="
echo "请下载这个文件到自己电脑，再用 CPU 测测试集："
echo "  $(pwd)/runs/cloud-20k/checkpoint.pt"
ls -lh runs/cloud-20k/checkpoint.pt
