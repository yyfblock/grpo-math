#!/usr/bin/env bash
# Stage 2-D：训推一致性直接测量
# 用法: bash scripts/run_tic.sh [题目数] [温度] [输出目录]
#   bash scripts/run_tic.sh 200 0.0 results/tic      贪心（默认）
#   bash scripts/run_tic.sh 200 1.0 results/tic_t1   复现训练时的采样条件
#
# 每阶段单独起进程（vLLM 与 HF 不能同时占显存），并且：
#   · 幂等 —— 产物已存在则跳过，可反复重跑
#   · 按【产物文件】判断成败，不看退出码 —— vLLM 进程常在清理阶段 abort，
#     但那时工作已完成、文件已写盘，看退出码会误判为失败。
set -u
cd "$(dirname "$0")/.."
source /etc/profile.d/conda.sh && conda activate rlpt
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${TIC_DEVICE:-0}"
export TOKENIZERS_PARALLELISM=false
N="${1:-200}"; T="${2:-0.0}"; D="${3:-results/tic}"
CK="${CKPT_ROOT:-./ckpt}"

step () {
  local out="$1"; local desc="$2"; shift 2
  if [ -e "$out" ]; then echo "--- 跳过（已存在）: $desc"; return 0; fi
  echo "=== $desc"
  "$@" || true
  if [ -e "$out" ]; then echo "    ok -> $out"; else echo "    !! 失败，产物不存在: $out"; return 1; fi
}

echo "### n=$N  temperature=$T  outdir=$D"
step "$CK/s2a1_merged/config.json" "0/6 合并 A1 adapter" \
     python src/merge_adapter.py "$CK/s2a1_lora_lr1e-5/final" "$CK/s2a1_merged"
step "$D/tokens.jsonl"     "1/6 固定 token 序列（T=$T, n=$N）" python src/tic_measure.py --phase gen  --n "$N" --temperature "$T" --outdir "$D"
step "$D/vllm_lora.json"   "2/6 vLLM 打分：挂 LoRA"  python src/tic_measure.py --phase vllm --mode lora   --outdir "$D"
step "$D/vllm_merged.json" "3/6 vLLM 打分：合并权重" python src/tic_measure.py --phase vllm --mode merged --outdir "$D"
step "$D/hf_lora.json"     "4/6 HF 打分：挂 LoRA"    python src/tic_measure.py --phase hf   --mode lora   --outdir "$D"
step "$D/hf_merged.json"   "5/6 HF 打分：合并权重"   python src/tic_measure.py --phase hf   --mode merged --outdir "$D"
echo "=== 6/6 汇总"
python src/tic_measure.py --phase report --outdir "$D"
