# -*- coding: utf-8 -*-
"""Stage 2-D：训推一致性（train-inference consistency）的直接测量。

问题
----
GRPO 训练时，采样端（vLLM）与训练端（HF transformers）对同一串 token 给出的
logprob 并不相同。训练日志里 TRL 记录的 sampling/importance_sampling_ratio
在 A1（LoRA）上均值 0.89、最坏 token 逼近 0；而 A2（全参）上均值 0.98、
最坏 token 在 0.35–0.47 —— 相差一个数量级。

假设：vLLM 加载 LoRA 时走独立的 LoRA 算子路径，与训练端 PEFT 的实现数值不同；
      全参微调没有这条分支，两边都用合并后的完整权重，所以一致性好得多。

验证方式
--------
同一个 adapter，构造两个【数学上完全等价】的模型：
    lora   = base + adapter（vLLM 走 LoRA 路径）
    merged = merge_and_unload 后的完整权重（vLLM 走普通路径）
对【同一串 token】分别用 vLLM 和 HF 打分，比较两种模式下的偏差。
若 lora 模式的偏差显著大于 merged 模式，则机制成立。

关键：token 序列必须固定。先贪心生成一次存下来，后续四次打分都用这一串，
      否则比较的就不是数值路径，而是"生成了不同的东西"。

用法（见 scripts/run_tic.sh，逐阶段单独起进程，避免 vLLM 与 HF 抢显存）
    python src/tic_measure.py --phase gen                      # 生成并固定 token
    python src/tic_measure.py --phase vllm --mode lora
    python src/tic_measure.py --phase vllm --mode merged
    python src/tic_measure.py --phase hf   --mode lora
    python src/tic_measure.py --phase hf   --mode merged
    python src/tic_measure.py --phase report
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cfg import load                      # noqa: E402
from data import load_split               # noqa: E402

BASE = 'Qwen/Qwen2.5-1.5B'
CKPT_ROOT = os.environ.get('CKPT_ROOT', 'ckpt')
ADAPTER = os.path.join(CKPT_ROOT, 's2a1_lora_lr1e-5', 'final')
MERGED = os.path.join(CKPT_ROOT, 's2a1_merged')
OUTDIR = 'results/tic'


def path(name):
    return os.path.join(OUTDIR, name)


# ------------------------------------------------------------------
# 阶段 1：固定 token 序列
# ------------------------------------------------------------------
def phase_gen(cfg, n_prompts, max_new, temperature):
    """用 merged 模型贪心生成一次，把 (prompt, 生成的 token_ids) 固定下来。

    贪心（temperature=0）是为了可复现；用 merged 而非 lora 生成，是因为
    merged 是"无额外算子路径"的参照系。
    """
    from vllm import LLM, SamplingParams
    os.makedirs(OUTDIR, exist_ok=True)
    items = load_split(cfg, cfg.data.test_split, n=n_prompts)
    prompts = [it['prompt'] for it in items]

    llm = LLM(model=MERGED, dtype=cfg.model.dtype,
              max_model_len=cfg.model.max_model_len,
              gpu_memory_utilization=cfg.gpu.vllm_mem_util)
    sp = SamplingParams(temperature=temperature, max_tokens=max_new,
                        logprobs=0, seed=1234)   # 固定 seed，T>0 时仍可复现
    outs = llm.generate(prompts, sp)

    with open(path('tokens.jsonl'), 'w', encoding='utf-8') as f:
        for it, o in zip(items, outs):
            c = o.outputs[0]
            # 生成路径的 logprob：vLLM 增量解码时给出的值，正是训练中 TRL
            # 用来算 importance_sampling_ratio 的那一份。与 prompt_logprobs
            # （teacher forcing 重新打分）可能不同，两者之差正是本阶段要测的。
            gen_lp = []
            for _i, _tid in enumerate(c.token_ids):
                _d = c.logprobs[_i] if c.logprobs and _i < len(c.logprobs) else None
                gen_lp.append(_d[_tid].logprob if _d and _tid in _d else None)
            f.write(json.dumps({
                'prompt': it['prompt'],
                'gold': it['gold'],
                'prompt_ids': list(o.prompt_token_ids),
                'completion_ids': list(c.token_ids),
                'gen_logprobs': gen_lp,
                'text': c.text,
            }, ensure_ascii=False) + '\n')
    print('已固定 %d 条样本（temperature=%.2f） -> %s'
          % (len(items), temperature, path('tokens.jsonl')))
    sys.stdout.flush()
    os._exit(0)


def read_tokens():
    rows = []
    with open(path('tokens.jsonl'), encoding='utf-8') as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


# ------------------------------------------------------------------
# 阶段 2：vLLM 打分（teacher forcing，用 prompt_logprobs）
# ------------------------------------------------------------------
def phase_vllm(cfg, mode):
    """把 prompt+completion 整串作为 prompt 喂进去，用 prompt_logprobs 取每个
    token 的 logprob —— 这样打的是【给定序列】的分，不是重新生成。
    """
    from vllm import LLM, SamplingParams
    rows = read_tokens()

    if mode == 'lora':
        from vllm.lora.request import LoRARequest
        llm = LLM(model=BASE, dtype=cfg.model.dtype,
                  max_model_len=cfg.model.max_model_len,
                  gpu_memory_utilization=cfg.gpu.vllm_mem_util,
                  enable_lora=True, max_lora_rank=32)
        lora_req = LoRARequest('a1', 1, ADAPTER)
    else:
        llm = LLM(model=MERGED, dtype=cfg.model.dtype,
                  max_model_len=cfg.model.max_model_len,
                  gpu_memory_utilization=cfg.gpu.vllm_mem_util)
        lora_req = None

    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    full = [{'prompt_token_ids': r['prompt_ids'] + r['completion_ids']} for r in rows]
    kw = {'lora_request': lora_req} if lora_req else {}
    outs = llm.generate(full, sp, **kw)

    res = []
    for r, o in zip(rows, outs):
        np_ = len(r['prompt_ids'])
        lp = o.prompt_logprobs           # 第 0 个恒为 None（没有前文）
        vals = []
        for i, tid in enumerate(r['completion_ids']):
            pos = np_ + i                # 该 token 在整串里的下标
            d = lp[pos] if pos < len(lp) else None
            vals.append(d[tid].logprob if d and tid in d else None)
        res.append(vals)

    with open(path('vllm_%s.json' % mode), 'w') as f:
        json.dump(res, f)
    ok = sum(1 for v in res for x in v if x is not None)
    print('vLLM[%s] 打分完成，有效 token %d 个' % (mode, ok))
    # vLLM 的引擎清理与解释器终结有竞争，退出时会 abort（活已干完、文件已写盘）。
    # 直接 _exit 绕过 finalization，避免驱动脚本误判为失败。
    sys.stdout.flush()
    os._exit(0)


# ------------------------------------------------------------------
# 阶段 3：HF 打分（训练端的路径）
# ------------------------------------------------------------------
def phase_hf(cfg, mode):
    import torch
    from transformers import AutoModelForCausalLM
    rows = read_tokens()

    dtype = torch.bfloat16
    if mode == 'lora':
        from peft import PeftModel
        m = AutoModelForCausalLM.from_pretrained(BASE, dtype=dtype).cuda()
        m = PeftModel.from_pretrained(m, ADAPTER)
    else:
        m = AutoModelForCausalLM.from_pretrained(MERGED, dtype=dtype).cuda()
    m.eval()

    res = []
    with torch.no_grad():
        for r in rows:
            ids = r['prompt_ids'] + r['completion_ids']
            x = torch.tensor([ids], device='cuda')
            logits = m(x).logits.float()          # 与训练端一致：bf16 前向、float 取对数
            lsm = torch.log_softmax(logits, dim=-1)[0]
            np_ = len(r['prompt_ids'])
            vals = []
            for i, tid in enumerate(r['completion_ids']):
                # 预测第 pos 个 token 的分布来自第 pos-1 个位置
                vals.append(float(lsm[np_ + i - 1, tid]))
            res.append(vals)

    with open(path('hf_%s.json' % mode), 'w') as f:
        json.dump(res, f)
    print('HF[%s] 打分完成' % mode)


# ------------------------------------------------------------------
# 阶段 4：比较
# ------------------------------------------------------------------
def summarize(V, H, label):
    diffs, ratios = [], []
    for a, b in zip(V, H):
        for x, y in zip(a, b):
            if x is None or y is None:
                continue
            d = y - x                     # 训练端 − 采样端
            diffs.append(abs(d))
            ratios.append(math.exp(max(-20.0, min(20.0, d))))
    if not ratios:
        print('%s: 无有效 token' % label)
        return None
    ratios.sort(); diffs.sort()
    n = len(ratios)
    q = lambda v, p: v[min(n - 1, int(n * p))]
    out = dict(
        label=label, n=n,
        ratio_mean=sum(ratios) / n, ratio_p01=q(ratios, .01), ratio_p50=q(ratios, .50),
        ratio_p99=q(ratios, .99), ratio_min=ratios[0], ratio_max=ratios[-1],
        absdiff_mean=sum(diffs) / n, absdiff_p99=q(diffs, .99), absdiff_max=diffs[-1],
        frac_out_20pct=sum(1 for r in ratios if r < 0.8 or r > 1.2) / n,
        frac_out_2x=sum(1 for r in ratios if r < 0.5 or r > 2.0) / n,
    )
    return out


def phase_report():
    def jload(fp):
        with open(fp) as f:
            return json.load(f)

    rows = []
    for mode in ('lora', 'merged'):
        vf, hf = path('vllm_%s.json' % mode), path('hf_%s.json' % mode)
        if os.path.exists(vf) and os.path.exists(hf):
            r = summarize(jload(vf), jload(hf), 'TF/%s' % mode)
            if r:
                rows.append(r)
    tf, hm = path('tokens.jsonl'), path('hf_merged.json')
    if os.path.exists(tf) and os.path.exists(hm):
        gen = [r.get('gen_logprobs') for r in read_tokens()]
        if any(g for g in gen):
            r = summarize(gen, jload(hm), 'GEN/merged')
            if r:
                rows.append(r)

    if not rows:
        print('没有可比较的结果'); return

    print('=' * 86)
    print('训推一致性：同一 adapter，两种加载路径（数学上等价的模型）')
    print('=' * 86)
    print('%-8s %8s %9s %9s %9s %9s %11s %10s' %
          ('模式', 'token数', 'ratio均值', 'ratio_p01', 'ratio_p99',
           'ratio最差', '|Δlogp|均值', '偏离>20%'))
    for r in rows:
        worst = min(abs(math.log(max(r['ratio_min'], 1e-12))),
                    abs(math.log(max(r['ratio_max'], 1e-12))))
        worst = r['ratio_min'] if abs(math.log(max(r['ratio_min'], 1e-12))) > \
            abs(math.log(max(r['ratio_max'], 1e-12))) else r['ratio_max']
        print('%-8s %8d %9.4f %9.4f %9.4f %9.4f %11.5f %9.2f%%' %
              (r['label'], r['n'], r['ratio_mean'], r['ratio_p01'], r['ratio_p99'],
               worst, r['absdiff_mean'], 100 * r['frac_out_20pct']))
    print()
    def byname(n):
        return next((r for r in rows if r['label'] == n), None)
    _tf, _gen = byname('TF/merged'), byname('GEN/merged')
    if _tf and _gen:
        print('★ 生成路径 vs teacher-forcing（同一模型、同一串 token）')
        print('   teacher-forcing 打分   |Δlogp| = %.5f   偏离>20%% = %.2f%%'
              % (_tf['absdiff_mean'], 100 * _tf['frac_out_20pct']))
        print('   生成路径打分           |Δlogp| = %.5f   偏离>20%% = %.2f%%'
              % (_gen['absdiff_mean'], 100 * _gen['frac_out_20pct']))
        print('   → 生成路径是 teacher-forcing 的 %.2f 倍'
              % (_gen['absdiff_mean'] / max(_tf['absdiff_mean'], 1e-12)))
        print()
    if False:
        a = byname('TF/lora'); b = byname('TF/merged')
        print('结论对照：')
        print('  |Δlogp| 均值    LoRA %.5f  vs  合并 %.5f   →  LoRA 是其 %.1f 倍'
              % (a['absdiff_mean'], b['absdiff_mean'],
                 a['absdiff_mean'] / max(b['absdiff_mean'], 1e-12)))
        print('  偏离超 20%%     LoRA %.2f%%  vs  合并 %.2f%%'
              % (100 * a['frac_out_20pct'], 100 * b['frac_out_20pct']))
        print('  偏离超 2 倍     LoRA %.2f%%  vs  合并 %.2f%%'
              % (100 * a['frac_out_2x'], 100 * b['frac_out_2x']))
        print()
        if a['absdiff_mean'] > 3 * b['absdiff_mean']:
            print('  → LoRA 加载路径是训推不一致的主要来源，假设成立。')
            print('    工程建议：训练用 LoRA，采样服务加载合并后的权重。')
        elif b['absdiff_mean'] > 3 * a['absdiff_mean']:
            print('  → 方向与假设相反，需要重新检查。')
        else:
            print('  → 两者量级相近，LoRA 路径不是主要来源，假设不成立。')

    with open(path('report.json'), 'w') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print('\n明细已写入 %s' % path('report.json'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True,
                    choices=['gen', 'vllm', 'hf', 'report'])
    ap.add_argument('--mode', default='merged', choices=['lora', 'merged'])
    ap.add_argument('--config', default='configs/base.yaml')
    ap.add_argument('--n', type=int, default=200, help='取多少道题')
    ap.add_argument('--max-new', type=int, default=400)
    ap.add_argument('--outdir', default='results/tic')
    ap.add_argument('--temperature', type=float, default=0.0,
                    help='生成阶段的采样温度。0=贪心；训练时用的是 1.0')
    args = ap.parse_args()

    global OUTDIR
    OUTDIR = args.outdir
    cfg = load(args.config)
    os.makedirs(OUTDIR, exist_ok=True)
    if args.phase == 'gen':
        phase_gen(cfg, args.n, args.max_new, args.temperature)
    elif args.phase == 'vllm':
        phase_vllm(cfg, args.mode)
    elif args.phase == 'hf':
        phase_hf(cfg, args.mode)
    else:
        phase_report()


if __name__ == '__main__':
    main()
