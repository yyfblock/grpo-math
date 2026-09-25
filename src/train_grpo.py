# -*- coding: utf-8 -*-
"""GRPO 训练入口。

设计说明（面试会问）：
  1. 三个奖励函数分开注册，TRL 会按函数名分别记录 ——
     这样能看出模型涨的是"格式"还是"答对"，而不是只看一个总分。
  2. 用 DiagCallback 额外记录 GRPO 特有的健康指标：
       - 零方差组占比：一组 G 条回答全对或全错 -> advantage 归零 -> 这批采样白烧
       - 长度分位数：p95 暴涨 = 模型在灌水（length hacking）
       - 关键词出现率：量化"模型什么时候开始写 step by step / Wait"
  3. 采样后端可切换（server / colocate / off），因为 TRL 1.13 官方只支持
     vLLM <= 0.28，而环境里是 0.29，兼容性有风险。
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                              # noqa: E402
from datasets import Dataset                    # noqa: E402
from transformers import TrainerCallback        # noqa: E402

from src.cfg import load                        # noqa: E402
from src.data import load_split                 # noqa: E402
from src.rewards import format_ok, extract_pred, num_equal  # noqa: E402
from src.eval import KEYWORDS                   # noqa: E402

# 每步的诊断信息由奖励函数写入，callback 读出后记录
_BUF = {}


def _completion_text(c):
    """TRL 的 completion 可能是 str，也可能是 [{'role':..,'content':..}]。"""
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c and isinstance(c[-1], dict):
        return c[-1].get("content", "")
    return str(c)


def make_reward_funcs(cfg):
    """返回三个独立的奖励函数。TRL 会分别记 rewards/<name>/mean。"""

    def r_format(completions, **kw):
        texts = [_completion_text(c) for c in completions]
        flags = [format_ok(t) for t in texts]
        _BUF["texts"] = texts
        _BUF["format"] = flags
        return [cfg.reward.w_format if f else 0.0 for f in flags]

    def r_answer(completions, gold=None, **kw):
        texts = [_completion_text(c) for c in completions]
        preds = [extract_pred(t) for t in texts]
        hits = [num_equal(p, g) for p, g in zip(preds, gold)]
        _BUF["correct"] = hits
        _BUF["gold"] = list(gold)
        return [cfg.reward.w_answer if h else 0.0 for h in hits]

    def r_length(completions, **kw):
        texts = [_completion_text(c) for c in completions]
        # 用字符数/4 粗估 token 数，避免在奖励函数里再 tokenize 一次
        approx = [len(t) // 4 for t in texts]
        _BUF["ntok"] = approx
        start, scale = cfg.reward.len_penalty_start, cfg.reward.len_penalty_scale
        return [-scale * max(0, n - start) for n in approx]

    return [r_format, r_answer, r_length]


class DiagCallback(TrainerCallback):
    """记录 GRPO 特有的健康指标 + 定期存样本快照。"""

    def __init__(self, cfg, G, outdir):
        self.G = G
        self.cfg = cfg
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.snap_every = 25

    def on_log(self, args, state, control, logs=None, **kw):
        if not _BUF.get("texts"):
            return
        texts = _BUF["texts"]
        correct = np.array(_BUF.get("correct", []), dtype=float)
        ntok = np.array(_BUF.get("ntok", [1]), dtype=float)
        fmt = np.array(_BUF.get("format", []), dtype=float)

        # --- 零方差组占比：GRPO 最重要的效率指标 ---
        zero_var = np.nan
        if len(correct) and len(correct) % self.G == 0:
            groups = correct.reshape(-1, self.G)
            zero_var = float(np.mean(groups.std(axis=1) < 1e-8))

        low = [t.lower() for t in texts]
        kwr = {f"emerge/{k}": float(np.mean([any(w in t for w in ws) for t in low]))
               for k, ws in KEYWORDS.items()}

        extra = {
            "diag/accuracy": float(correct.mean()) if len(correct) else np.nan,
            "diag/format_valid": float(fmt.mean()) if len(fmt) else np.nan,
            "diag/zero_var_group_ratio": zero_var,
            "diag/len_mean": float(ntok.mean()),
            "diag/len_p95": float(np.percentile(ntok, 95)),
            "diag/len_max": float(ntok.max()),
            **kwr,
        }
        if logs is not None:
            logs.update({k: v for k, v in extra.items() if v == v})  # 过滤 nan

        # --- 样本快照：训练时的真实输出，早上要人工读 ---
        if state.global_step % self.snap_every == 0:
            snap = self.outdir / f"train_samples_step{state.global_step}.jsonl"
            with open(snap, "w", encoding="utf-8") as f:
                for i, t in enumerate(texts[:16]):
                    f.write(json.dumps({
                        "step": state.global_step, "text": t,
                        "gold": _BUF.get("gold", [None] * len(texts))[i]
                        if i < len(_BUF.get("gold", [])) else None,
                        "correct": int(correct[i]) if i < len(correct) else None,
                        "format_ok": int(fmt[i]) if i < len(fmt) else None,
                    }, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--train-config", default="configs/grpo_1.5b.yaml")
    a = ap.parse_args()
    cfg = load(a.config, a.train_config)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg.gpu.train_device))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from trl import GRPOConfig, GRPOTrainer

    # ---------- 数据 ----------
    rows = load_split(cfg, cfg.data.train_split, seed=cfg.data.seed)
    ds = Dataset.from_list([{"prompt": r["prompt"], "gold": r["gold"]} for r in rows])
    print(f"[data] 训练集 {len(ds)} 条")

    t = cfg.train
    outdir = Path(cfg.paths.ckpt) / t.run_name
    resdir = Path(cfg.paths.results) / t.run_name
    resdir.mkdir(parents=True, exist_ok=True)

    vargs = {}
    mode = cfg.vllm.mode
    if mode in ("server", "colocate"):
        vargs.update(use_vllm=True, vllm_mode=mode)
        if mode == "server":
            vargs.update(vllm_server_host="127.0.0.1", vllm_server_port=cfg.vllm.port)
        else:
            vargs.update(vllm_gpu_memory_utilization=0.35)
    else:
        vargs.update(use_vllm=False)

    args = GRPOConfig(
        output_dir=str(outdir),
        run_name=t.run_name,
        max_steps=t.max_steps,
        num_generations=t.num_generations,
        per_device_train_batch_size=t.per_device_train_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        temperature=t.temperature,
        beta=t.beta,
        epsilon=t.epsilon,
        scale_rewards=t.scale_rewards,
        loss_type=t.loss_type,
        learning_rate=float(t.learning_rate),
        lr_scheduler_type=t.lr_scheduler_type,
        warmup_steps=t.warmup_steps,
        max_grad_norm=t.max_grad_norm,
        max_completion_length=t.max_completion_length,
        bf16=t.bf16,
        gradient_checkpointing=t.gradient_checkpointing,
        logging_steps=t.logging_steps,
        save_steps=t.save_steps,
        save_total_limit=t.get('save_total_limit', 3),
        # --- 熵控制（Stage 2-B 新增；旧配置不含这些键时用默认值，行为不变）---
        entropy_coef=t.get('entropy_coef', 0.0),
        use_adaptive_entropy=t.get('use_adaptive_entropy', False),
        entropy_target=t.get('entropy_target', 0.2),
        entropy_coef_max=t.get('entropy_coef_max', 1.0),
        entropy_coef_delta=t.get('entropy_coef_delta', 0.005),
        seed=t.seed,
        report_to=list(cfg.log.report_to),
        log_completions=True,
        **vargs,
    )

    peft_cfg = None
    if cfg.lora.enabled:
        from peft import LoraConfig
        peft_cfg = LoraConfig(
            r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout,
            target_modules=list(cfg.lora.target_modules),
            task_type="CAUSAL_LM",
        )
        print(f"[lora] r={cfg.lora.r} alpha={cfg.lora.alpha}")

    trainer = GRPOTrainer(
        model=cfg.model.name,
        reward_funcs=make_reward_funcs(cfg),
        args=args,
        train_dataset=ds,
        peft_config=peft_cfg,
        callbacks=[DiagCallback(cfg, t.num_generations, resdir)],
    )
    print(f"[train] mode={mode}  G={t.num_generations}  steps={t.max_steps}")
    trainer.train()
    final = outdir / "final"
    trainer.save_model(str(final))
    try:
        trainer.processing_class.save_pretrained(str(final))
    except Exception as e:
        print("[warn] tokenizer 未保存:", e)
    print("[done] saved ->", final)


if __name__ == "__main__":
    main()
