# -*- coding: utf-8 -*-
"""奖励函数 —— 整个项目里唯一注入先验的地方，也最容易出 bug。

三层设计（分开记录，才知道模型涨的是格式还是能力）：
  1. format  : 输出是否符合 <think>...</think><answer>...</answer> 结构
  2. answer  : <answer> 里的数值是否等于标准答案
  3. length  : 超长惩罚，防止模型靠灌水凑奖励（length hacking）

注意：prompt 末尾已经给了 "<think>"，所以模型的补全从 think 内容开始，
     不需要再输出开头的 <think> 标签。
"""
import re

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_pred(completion: str):
    """从补全里抽出预测答案。抽不到返回 None。"""
    m = ANSWER_RE.search(completion)
    if not m:
        return None
    body = m.group(1).replace(",", "").replace("$", "").strip()
    nums = NUM_RE.findall(body)
    return nums[-1] if nums else None


def format_ok(completion: str) -> bool:
    """结构合法：先闭合 think，再有完整的 answer 段。"""
    if "</think>" not in completion:
        return False
    if not ANSWER_RE.search(completion):
        return False
    # answer 必须出现在 think 闭合之后
    return completion.index("</think>") < completion.index("<answer>")


def num_equal(a, b, tol=1e-4) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < tol
    except (TypeError, ValueError):
        return str(a).strip() == str(b).strip()


def score_one(cfg, completion: str, gold: str, n_tokens: int) -> dict:
    """单条打分，返回各分项（便于分开记录到 TensorBoard）。"""
    fmt = format_ok(completion)
    pred = extract_pred(completion)
    correct = num_equal(pred, gold)

    r_format = cfg.reward.w_format if fmt else 0.0
    r_answer = cfg.reward.w_answer if correct else 0.0

    over = max(0, n_tokens - cfg.reward.len_penalty_start)
    r_len = -cfg.reward.len_penalty_scale * over

    return {
        "reward": r_format + r_answer + r_len,
        "r_format": r_format,
        "r_answer": r_answer,
        "r_len": r_len,
        "format_ok": int(fmt),
        "correct": int(correct),
        "pred": pred,
        "n_tokens": n_tokens,
    }


# ---------------------------------------------------------------
# 自测：奖励函数写完必须像单元测试一样手工验证
# ---------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.cfg import load

    cfg = load(os.path.join(os.path.dirname(__file__), "..", "configs", "base.yaml"))

    CASES = [
        # (补全, 标准答案, 期望 format_ok, 期望 correct)
        ("reasoning here</think><answer>72</answer>", "72", 1, 1),
        ("reasoning</think>\n<answer> 72 </answer>", "72", 1, 1),
        ("reasoning</think><answer>The answer is 72</answer>", "72", 1, 1),
        ("reasoning</think><answer>1,234</answer>", "1234", 1, 1),
        ("reasoning</think><answer>36</answer>", "72", 1, 0),   # 格式对答案错
        ("<answer>72</answer>", "72", 0, 1),                    # 没闭合 think
        ("reasoning</think> the answer is 72", "72", 0, 0),     # 没有 answer 标签
        ("<answer>72</answer> more</think>", "72", 0, 1),       # 顺序错
        ("reasoning</think><answer></answer>", "72", 1, 0),     # 空 answer
        ("reasoning</think><answer>72.0</answer>", "72", 1, 1),  # 浮点等价
    ]
    bad = 0
    for comp, gold, ef, ec in CASES:
        s = score_one(cfg, comp, gold, n_tokens=100)
        ok = (s["format_ok"] == ef) and (s["correct"] == ec)
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'}  fmt={s['format_ok']}(want {ef})  "
              f"correct={s['correct']}(want {ec})  pred={s['pred']!r}  {comp[:46]!r}")
    print(f"\n长度惩罚检查: 500 tok -> {score_one(cfg,'a</think><answer>1</answer>','1',500)['r_len']:.3f}"
          f" | 900 tok -> {score_one(cfg,'a</think><answer>1</answer>','1',900)['r_len']:.3f}")
    print(f"\n{'全部通过' if bad == 0 else str(bad) + ' 条失败'}")
    sys.exit(1 if bad else 0)
