# -*- coding: utf-8 -*-
"""实验 A：SFT 冷启动训练入口。

产出一个 LoRA adapter，作为后续 GRPO 的起点（测点 AC），
同时它本身也要单独评测一次（测点 S）—— 不测 S 就拆不开 SFT 与 RL 各自的贡献。

用法：
    bash scripts/run_sft.sh                       # 用 configs/sft_1.5b.yaml
    bash scripts/run_sft.sh configs/其他.yaml
"""
import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cfg import load                      # noqa: E402


def filter_known(kwargs, dc, label):
    """把配置项对着目标 dataclass 的真实字段全量核验。

    Stage 1 的教训：前三次是一个参数一个参数试错，第四次改成一次性对着
    dataclasses.fields() 全量核验，一次通过。这里把那个做法固化下来 ——
    TRL 版本间字段增删频繁，静默丢参数比直接报错危险得多。
    """
    known = {f.name for f in dataclasses.fields(dc)}
    good = {k: v for k, v in kwargs.items() if k in known}
    bad = sorted(set(kwargs) - known)
    if bad:
        print('!! %s 不接受以下字段，已拒绝启动：%s' % (label, ', '.join(bad)))
        print('   该版本共有 %d 个合法字段。请对照后修改配置。' % len(known))
        sys.exit(2)
    return good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/base.yaml')
    ap.add_argument('--train-config', default='configs/sft_1.5b.yaml')
    args = ap.parse_args()

    cfg = load(args.config, args.train_config)
    sft = dict(cfg.sft)
    run_name = sft.pop('run_name')
    data_file = sft.pop('data_file')

    out_dir = os.path.join(cfg.paths.ckpt, run_name)
    os.makedirs(out_dir, exist_ok=True)

    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    # ---- 数据 ----
    rows = []
    with open(data_file, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            rows.append({'prompt': r['prompt'], 'completion': r['completion']})
    if not rows:
        print('!! %s 为空，先跑 python src/build_sft_data.py' % data_file)
        sys.exit(2)
    ds = Dataset.from_list(rows)
    print('训练样本 %d 条' % len(ds))
    print('样例 prompt 末尾：%r' % rows[0]['prompt'][-30:])
    print('样例 completion ：%r' % rows[0]['completion'][:80])

    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- 训练参数（全量核验后再传） ----
    sft.update({
        'output_dir': out_dir,
        'run_name': run_name,
        'report_to': ['tensorboard'],
    })
    sft_args = SFTConfig(**filter_known(sft, SFTConfig, 'SFTConfig'))

    # ---- LoRA：与 A1 完全一致 ----
    peft_cfg = None
    if cfg.lora.enabled:
        from peft import LoraConfig
        peft_cfg = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=list(cfg.lora.target_modules),
            bias='none',
            task_type='CAUSAL_LM',
        )
        # 必须打印实际挂载的模块：不打印就看不出 PEFT 用了默认值（只挂 q/v 两个），
        # 这正是 2026-09-24 首轮 SFT 与 A1 不可比的原因。
        print('LoRA r=%d alpha=%d dropout=%s' % (cfg.lora.r, cfg.lora.alpha, cfg.lora.dropout))
        print('LoRA target_modules(%d) = %s'
              % (len(cfg.lora.target_modules), list(cfg.lora.target_modules)))

    trainer = SFTTrainer(
        model=cfg.model.name,
        args=sft_args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(out_dir)
    print('已保存：%s' % out_dir)
    print('下一步（测点 S）：bash scripts/run_eval.sh s_sft_only --lora %s' % out_dir)


if __name__ == '__main__':
    main()
