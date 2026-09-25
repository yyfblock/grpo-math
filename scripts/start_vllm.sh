#!/usr/bin/env bash
# 在 GPU0 上拉起 TRL 的 vLLM 采样服务
# 用法: bash scripts/start_vllm.sh [模型路径]
#   不传参数 → Qwen/Qwen2.5-1.5B，行为与 2026-09-25 之前完全一致
#   传参数   → 用指定模型（测点 AC 用合并后的 SFT 模型）
# 显存占比可用环境变量 VLLM_MEM 覆盖（共享机器上他人占卡时需调低）
set -e
cd "$(dirname "$0")/.."
source /etc/profile.d/conda.sh && conda activate rlpt
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
MODEL="${1:-Qwen/Qwen2.5-1.5B}"
VLLM_MEM="${VLLM_MEM:-0.80}"
echo "[vllm] model=$MODEL  mem=$VLLM_MEM"
exec trl vllm-serve --model "$MODEL" \
    --port 8765 --gpu-memory-utilization "$VLLM_MEM" --max-model-len 1536
