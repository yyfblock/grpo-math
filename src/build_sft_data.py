# -*- coding: utf-8 -*-
"""实验 A：把 GSM8K 训练集自带的人工解题过程转成 SFT 训练样本。

数据零采购成本 —— GSM8K 的 answer 字段本身就是人工撰写的分步解题过程，
只需剥离计算器标注、补上闭合标签即可。

★ 设计要点：prompt 直接复用 src/data.py::build_prompt()，不在本文件里重抄模板。
  SFT 的 prompt 必须与 GRPO 阶段逐字一致，否则冷启动模型进 GRPO 会分布错配，
  实验就不是单变量的了。复用函数是唯一能保证不漂移的做法。

★ 输出格式：base.yaml 的 prompt 模板以 "Assistant: <think>" 结尾，
  模型的输出是从 think 块【内部】开始的，所以 completion 只需闭合它，
  不能再带开标签 —— 这也是 rewards.py::format_ok() 只检查 "</think>"
  而不检查 "<think>" 的原因。

用法（在仓库根目录执行）：
    python src/build_sft_data.py --self-test      # 先跑自测
    python src/build_sft_data.py                  # 生成 data/sft_gsm8k.jsonl
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cfg import load                      # noqa: E402
from data import build_prompt             # noqa: E402

# completion 格式：闭合 think，再给出答案
COMPLETION_TEMPLATE = '{reasoning}\n</think>\n<answer>{answer}</answer>'

# GSM8K 计算器标注，形如 <<48/2=24>>
CALC_RE = re.compile(r'<<[^>]*>>')
# GSM8K 最终答案行，形如 "#### 72"
FINAL_RE = re.compile(r'####\s*(.+?)\s*$', re.MULTILINE)


def strip_calculator(text):
    """剥离 <<...>> 计算器标注并清理多余空白。

    那是 GSM8K 为训练辅助加的标记，不是要让模型学会输出的内容。
    """
    text = CALC_RE.sub('', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r' +\n', '\n', text)
    return text.strip()


def split_answer(raw):
    """把 GSM8K 的 answer 字段拆成 (推理过程, 最终答案)。

    返回 (None, None) 表示无法解析，应当丢弃而不是猜。
    """
    if not raw:
        return None, None
    m = FINAL_RE.search(raw)
    if not m:
        return None, None
    # 与 data.py::gold_answer 的口径一致：去掉千分位逗号
    final = m.group(1).strip().replace(',', '')
    reasoning = strip_calculator(raw[:m.start()])
    if not reasoning or not final:
        return None, None
    return reasoning, final


def build_example(cfg, question, raw_answer):
    """构造一条 SFT 样本；无法解析时返回 None。"""
    reasoning, final = split_answer(raw_answer)
    if reasoning is None:
        return None
    return {
        'prompt': build_prompt(cfg, question),          # ← 复用 GRPO 的同一函数
        'completion': COMPLETION_TEMPLATE.format(reasoning=reasoning, answer=final),
        'gold': final,
    }


# ----------------------------- 自测 -----------------------------
# 与 rewards.py 的做法一致：格式相关的代码必须像单元测试一样验证。
_CASES = [
    ('标准两步',
     'Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n'
     'Natalia sold 48+24 = <<48+24=72>>72 clips altogether.\n#### 72', '72'),
    ('带逗号数字',
     'He earns 1,000 * 2 = <<1000*2=2000>>2,000 dollars.\n#### 2,000', '2000'),
    ('单步', 'She has 5 apples.\n#### 5', '5'),
    ('多个计算器标注',
     'a = 3*4 = <<3*4=12>>12\nb = 12+8 = <<12+8=20>>20\n#### 20', '20'),
    ('小数答案', 'The average is 7/2 = <<7/2=3.5>>3.5.\n#### 3.5', '3.5'),
    ('负数答案', 'The change is 5-8 = <<5-8=-3>>-3.\n#### -3', '-3'),
    ('缺 #### 行 → 丢弃', 'Some reasoning without a final marker.', None),
    ('空字符串 → 丢弃', '', None),
    ('只有 #### 没有推理 → 丢弃', '#### 42', None),
    ('#### 后为空 → 丢弃', 'Reasoning here.\n#### ', None),
]


def self_test(cfg):
    from rewards import format_ok          # 用项目里的真实实现，不复刻
    ok = True
    for name, raw, expect in _CASES:
        ex = build_example(cfg, 'dummy question', raw)
        got = ex['gold'] if ex else None
        passed = (got == expect)
        note = ''
        if ex and not format_ok(ex['completion']):
            passed, note = False, '  ← format_ok 未通过'
        ok &= passed
        print('%s  %-22s 期望=%-6s 实际=%-6s%s'
              % ('PASS' if passed else 'FAIL', name, expect, got, note))

    ex = build_example(cfg, 'q', 'x = 2*3 = <<2*3=6>>6\n#### 6')
    clean = '<<' not in ex['completion'] and '>>' not in ex['completion']
    ok &= clean
    print('%s  计算器标注已剥离' % ('PASS' if clean else 'FAIL'))

    # prompt 必须与 GRPO 阶段完全一致
    same = ex['prompt'] == build_prompt(cfg, 'q')
    ok &= same
    print('%s  prompt 与 build_prompt() 一致' % ('PASS' if same else 'FAIL'))

    # completion 不得包含开标签（prompt 里已经有了）
    no_open = '<think>' not in ex['completion']
    ok &= no_open
    print('%s  completion 不含多余的 <think> 开标签' % ('PASS' if no_open else 'FAIL'))

    print('\n样例：')
    print('  prompt 末尾 ……%r' % ex['prompt'][-30:])
    print('  completion  %r' % ex['completion'])
    print('\n自测结果：%s' % ('全部通过' if ok else '存在失败项'))
    return 0 if ok else 1


# ----------------------------- 主流程 -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--config', default='configs/base.yaml')
    ap.add_argument('--out', default='data/sft_gsm8k.jsonl')
    ap.add_argument('--limit', type=int, default=0, help='只取前 N 条，0 表示全部')
    args = ap.parse_args()

    cfg = load(args.config)

    if args.self_test:
        sys.exit(self_test(cfg))

    from datasets import load_dataset
    from rewards import format_ok
    ds = load_dataset(cfg.data.dataset, cfg.data.subset, split=cfg.data.train_split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    kept = dropped = 0
    with open(args.out, 'w', encoding='utf-8') as f:
        for row in ds:
            ex = build_example(cfg, row['question'], row['answer'])
            if ex is None or not format_ok(ex['completion']):
                dropped += 1
                continue
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
            kept += 1

    print('写入 %s' % args.out)
    print('保留 %d 条，丢弃 %d 条（丢弃率 %.2f%%）'
          % (kept, dropped, 100.0 * dropped / max(1, kept + dropped)))
    if dropped > kept * 0.02:
        print('!! 丢弃率偏高，建议抽查原始样本，确认不是解析逻辑有误')


if __name__ == '__main__':
    main()
