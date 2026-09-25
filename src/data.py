# -*- coding: utf-8 -*-
"""GSM8K 加载 + prompt 构造 + 标准答案抽取。

GSM8K 的 answer 字段长这样：
    "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n#### 72"
最终答案永远在 '#### ' 之后。
"""
import re
from datasets import load_dataset

GOLD_RE = re.compile(r"####\s*([\-0-9\.,/]+)")


def gold_answer(ans_field: str) -> str:
    """从 GSM8K 的 answer 字段抽出最终数值答案。"""
    m = GOLD_RE.search(ans_field)
    if not m:
        raise ValueError(f"无法解析标准答案: {ans_field[-80:]!r}")
    return m.group(1).strip().replace(",", "")


def build_prompt(cfg, question: str) -> str:
    return cfg.prompt.template.format(
        system=cfg.prompt.system.strip(), question=question.strip()
    )


def load_split(cfg, split: str, n=None, seed=None):
    """返回 [{question, gold, prompt}, ...]"""
    ds = load_dataset(cfg.data.dataset, cfg.data.subset, split=split)
    if seed is not None:
        ds = ds.shuffle(seed=seed)
    if n:
        ds = ds.select(range(min(n, len(ds))))
    out = []
    for row in ds:
        out.append(
            {
                "question": row["question"],
                "gold": gold_answer(row["answer"]),
                "prompt": build_prompt(cfg, row["question"]),
            }
        )
    return out


def load_probes(cfg):
    """固定探针题：不参与训练，用于跨 checkpoint 并排对比。"""
    return load_split(cfg, cfg.data.test_split, n=cfg.probe.n, seed=cfg.probe.seed)
