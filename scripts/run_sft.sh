#!/usr/bin/env bash
# 用法: bash scripts/run_sft.sh [sft-config]
# 在 tmux 里跑：tmux new -d -s sft 'bash scripts/run_sft.sh > results/train_sft.log 2>&1'
set -e
cd "$(dirname "$0")/.."
source /etc/profile.d/conda.sh && conda activate rlpt
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# SFT 只用训练卡，采样卡留给 vLLM（与 run_train.sh 的分工一致）
export CUDA_VISIBLE_DEVICES="${SFT_DEVICE:-1}"
export TOKENIZERS_PARALLELISM=false
python src/train_sft.py --train-config "${1:-configs/sft_1.5b.yaml}"
