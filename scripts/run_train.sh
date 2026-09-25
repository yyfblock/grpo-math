#!/usr/bin/env bash
# 用法: bash scripts/run_train.sh [train-config]
# 在 tmux 里跑：tmux new -s train 'bash scripts/run_train.sh'
set -e
cd "$(dirname "$0")/.."
source /etc/profile.d/conda.sh && conda activate rlpt
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
python src/train_grpo.py --train-config "${1:-configs/grpo_1.5b.yaml}"
