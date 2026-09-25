#!/usr/bin/env bash
# Stage 2-B 全流程驱动：A1(LoRA+大lr) 与 A2(全参) 串行跑完并各自评测。
# 用法：tmux new -d -s s2a 'bash scripts/run_stage2a.sh'
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd)
CKPT="${CKPT_ROOT:-./ckpt}"
LOG=$ROOT/results

start_vllm () {
  tmux kill-session -t vllm 2>/dev/null
  sleep 3
  tmux new-session -d -s vllm "cd $ROOT && bash scripts/start_vllm.sh > $LOG/vllm_$1.log 2>&1"
  echo "[driver] 等待 vLLM 起来 ..."
  for _ in $(seq 1 60); do
    if grep -q "Application startup complete" "$LOG/vllm_$1.log" 2>/dev/null; then
      echo "[driver] vLLM ready"; return 0
    fi
    sleep 10
  done
  echo "[driver] !! vLLM 启动超时"; return 1
}

stop_vllm () { tmux kill-session -t vllm 2>/dev/null; sleep 5; }

run_one () {                       # $1=tag  $2=train-config  $3=lora?
  local tag=$1 conf=$2 is_lora=$3
  echo "=============================================================="
  echo "[driver] >>> 开始 $tag   config=$conf   $(date '+%F %T')"
  echo "=============================================================="
  start_vllm "$tag" || return 1
  bash scripts/run_train.sh "$conf" > "$LOG/train_$tag.log" 2>&1
  local rc=$?
  stop_vllm
  if [ $rc -ne 0 ]; then
    echo "[driver] !! $tag 训练失败 rc=$rc，日志 $LOG/train_$tag.log"
    tail -25 "$LOG/train_$tag.log"
    return 1
  fi
  echo "[driver] $tag 训练完成，开始评测 $(date '+%F %T')"
  local final=$CKPT/$tag/final
  if [ "$is_lora" = "lora" ]; then
    bash scripts/run_eval.sh "eval_$tag" --lora "$final" > "$LOG/eval_$tag.log" 2>&1
  else
    bash scripts/run_eval.sh "eval_$tag" --model "$final" > "$LOG/eval_$tag.log" 2>&1
  fi
  echo "[driver] $tag 评测完成"
  grep -A20 '^=====' "$LOG/eval_$tag.log" | head -24
}

echo "########## Stage 2-B 开始 $(date '+%F %T') ##########"
run_one s2b1_entropy_ctrl configs/grpo_b1_entropy.yaml lora
run_one s2b2_seed43      configs/grpo_b2_seed43.yaml  lora

echo
echo "########## 配对显著性检验 ##########"
source /etc/profile.d/conda.sh && conda activate rlpt
for t in eval_s2b1_entropy_ctrl eval_s2b2_seed43; do
  if [ -f "$LOG/$t.jsonl" ]; then
    echo; echo "### baseline vs $t"
    python src/compare.py --before baseline_base_1.5b --after "$t"
  fi
done
echo "########## Stage 2-B 结束 $(date '+%F %T') ##########"
