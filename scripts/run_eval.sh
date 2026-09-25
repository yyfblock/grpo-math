#!/usr/bin/env bash
# 用法: bash scripts/run_eval.sh <tag> [--probe] [额外参数]
set -e
cd "$(dirname "$0")/.."
source /etc/profile.d/conda.sh && conda activate rlpt
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
TAG="$1"; shift
python src/eval.py --tag "$TAG" "$@"
