# -*- coding: utf-8 -*-
"""评测：用 vLLM 批量生成，按 rewards.py 的口径打分。

关键原则：评测脚本、prompt 模板、采样参数必须和训练时完全一致，
         否则前后对比是假的。所以全部从同一份 config 读。
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.cfg import load                      # noqa: E402
from src.data import load_split, load_probes  # noqa: E402
from src.rewards import score_one             # noqa: E402

# 涌现行为关键词：量化"模型什么时候开始写 step by step"
KEYWORDS = {
    "stepwise":   ["step by step", "first,", "let me", "we need to"],
    "reflection": ["wait", "hmm", "actually", "recheck", "let me check again"],
    "verify":     ["verify", "check", "substitut", "makes sense"],
    "conclude":   ["therefore", "so the answer", "in total", "thus"],
}


def keyword_rates(texts):
    n = max(1, len(texts))
    low = [t.lower() for t in texts]
    return {k: sum(any(w in t for w in ws) for t in low) / n for k, ws in KEYWORDS.items()}


def run(cfg, model_path, tag, probe_only=False, lora=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu.vllm_device)
    from vllm import LLM, SamplingParams

    items = load_probes(cfg) if probe_only else load_split(
        cfg, cfg.data.test_split, n=cfg.data.eval_n)
    print(f"[eval] tag={tag}  model={model_path}  n={len(items)}  probe_only={probe_only}")

    llm = LLM(model=model_path, dtype=cfg.model.dtype,
              gpu_memory_utilization=cfg.gpu.vllm_mem_util,
              max_model_len=cfg.model.max_model_len,
              enable_lora=lora is not None, max_lora_rank=64 if lora else None)

    sp = SamplingParams(n=cfg.generation.n,
                        temperature=cfg.generation.temperature,
                        top_p=cfg.generation.top_p,
                        max_tokens=cfg.generation.max_tokens,
                        seed=cfg.probe.seed if probe_only else None,
                        stop=["</answer>", "\nUser:"],
                        include_stop_str_in_output=True)

    t0 = time.time()
    kw = {}
    if lora:
        from vllm.lora.request import LoRARequest
        kw["lora_request"] = LoRARequest("adapter", 1, lora)
    outs = llm.generate([it["prompt"] for it in items], sp, **kw)
    dt = time.time() - t0

    records, texts = [], []
    for it, o in zip(items, outs):
        comp = o.outputs[0].text
        s = score_one(cfg, comp, it["gold"], len(o.outputs[0].token_ids))
        records.append({"question": it["question"], "gold": it["gold"],
                        "completion": comp, **s})
        texts.append(comp)

    n = len(records)
    acc = sum(r["correct"] for r in records) / n
    fmt = sum(r["format_ok"] for r in records) / n
    lens = sorted(r["n_tokens"] for r in records)
    summary = {
        "tag": tag, "model": model_path, "n": n,
        "accuracy": round(acc, 4),
        "format_valid_ratio": round(fmt, 4),
        "reward_mean": round(sum(r["reward"] for r in records) / n, 4),
        "len_mean": round(sum(lens) / n, 1),
        "len_p50": lens[n // 2],
        "len_p95": lens[int(n * 0.95) - 1],
        "len_max": lens[-1],
        "gen_seconds": round(dt, 1),
        "keyword_rates": {k: round(v, 4) for k, v in keyword_rates(texts).items()},
        "pred_none_ratio": round(sum(r["pred"] is None for r in records) / n, 4),
    }

    outdir = Path(cfg.paths.probes if probe_only else cfg.paths.results)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / f"{tag}.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(outdir / f"{tag}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 62)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 62)
    print(f"明细: {outdir / (tag + '.jsonl')}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--model", default=None, help="默认用 config 里的基座")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--probe", action="store_true", help="只跑 12 道固定探针题")
    ap.add_argument("--lora", default=None)
    a = ap.parse_args()
    c = load(a.config)
    run(c, a.model or c.model.name, a.tag, probe_only=a.probe, lora=a.lora)
