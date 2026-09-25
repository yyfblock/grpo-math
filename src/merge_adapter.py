# -*- coding: utf-8 -*-
"""把 LoRA adapter 合并进基座，产出等价的完整权重模型。

用法：
    python src/merge_adapter.py <adapter路径> <输出路径> [基座]
幂等：输出目录已存在且含 config.json 则跳过。
"""
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    adapter, out = sys.argv[1], sys.argv[2]
    base = sys.argv[3] if len(sys.argv) > 3 else 'Qwen/Qwen2.5-1.5B'

    if os.path.exists(os.path.join(out, 'config.json')):
        print('已存在，跳过：', out)
        return

    print('基座   :', base)
    print('adapter:', adapter)
    m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    m = PeftModel.from_pretrained(m, adapter)
    m = m.merge_and_unload()
    m.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(out)
    print('已保存 :', out)


if __name__ == '__main__':
    main()
