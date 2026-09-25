# -*- coding: utf-8 -*-
"""两次评测的配对显著性检验（McNemar）。

为什么用 McNemar 而不是独立两样本比例检验：
两次评测跑的是同一批题目，属于配对数据。McNemar 只看"一个对一个错"
的不一致对，统计效力比把两组当独立样本高得多。
"""
import argparse
import json
import math
from math import erfc, sqrt
from pathlib import Path


def load(results_dir, tag):
    p = Path(results_dir) / f"{tag}.jsonl"
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def mcnemar(a_flags, b_flags):
    b01 = sum(1 for x, y in zip(a_flags, b_flags) if not x and y)
    b10 = sum(1 for x, y in zip(a_flags, b_flags) if x and not y)
    if b01 + b10 == 0:
        return b01, b10, float("nan"), 1.0
    chi2 = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)   # 连续性校正
    p = erfc(sqrt(chi2 / 2))
    return b01, b10, chi2, p


def report(a, b, field, name):
    fa = [bool(x[field]) for x in a]
    fb = [bool(x[field]) for x in b]
    n = len(fa)
    ra, rb = sum(fa) / n, sum(fb) / n
    b01, b10, chi2, p = mcnemar(fa, fb)
    se = math.sqrt(ra * (1 - ra) / n)
    verdict = "显著 (p<0.05)" if p < 0.05 else "不显著 —— 无法与噪声区分"
    print(f"\n--- {name} ---")
    print(f"  基线 {ra:.3f}  ->  训练后 {rb:.3f}   差值 {rb - ra:+.3f}")
    print(f"  新变好 {b01} 题 / 新变差 {b10} 题 / 净 {b01 - b10:+d}")
    print(f"  McNemar chi2={chi2:.3f}  p={p:.4g}  ->  {verdict}")
    print(f"  单点标准误 ±{se * 100:.1f} 个百分点")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    a = ap.parse_args()
    A, B = load(a.dir, a.before), load(a.dir, a.after)
    assert [x["question"] for x in A] == [x["question"] for x in B], \
        "两次评测的题目不一致，不能做配对检验"
    print(f"题目完全配对  n={len(A)}   {a.before}  vs  {a.after}")
    report(A, B, "correct", "答案正确率")
    report(A, B, "format_ok", "格式合规率")
