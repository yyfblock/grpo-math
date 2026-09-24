# grpo-math

> 用 GRPO 让一个未经微调的小模型自己学会做数学题 —— 不教解法，只告诉它答案对不对。

在 **Qwen2.5-1.5B Base** 上用 GRPO 做 GSM8K 数学推理后训练。
测试集准确率 **52.2% → 71.6%**（配对 McNemar 检验 *p* < 1e-13）。

这个仓库的重点不在那 19 个点，而在**怎么确认这 19 个点是真的**——
第一轮实验涨了 2.8 个点看起来很美，配对检验一做 *p*=0.32，是噪声。

---

## 核心结果

| 实验 | 准确率 | 格式合规 | vs 基线 | 判定 |
|---|---|---|---|---|
| 基线（未训练） | 0.522 | 0.852 | — | — |
| Stage 1（LoRA, lr=1e-6, 200 步） | 0.550 | 0.966 | +2.8 点，*p*=0.322 | ❌ **不显著** |
| Stage 2-A1（LoRA, lr=1e-5, 500 步） | **0.716** | 0.998 | +19.4 点，*p*=1.8e-14 | ✅ 极显著 |
| Stage 2-A2（全参, lr=2e-6, 500 步） | **0.720** | 0.996 | +19.8 点，*p*=1.3e-13 | ✅ 极显著 |

测试集为 GSM8K 全部 500 题，四组实验使用同一批题目、同一套奖励函数、同一随机种子。

**三个附带结论：**

1. **LoRA 不是瓶颈** —— LoRA(r=32) 与全参微调差 0.4 个点，小于单点标准误 2.2 个点，统计上不可区分。
2. **后一半训练是白跑的** —— 全部收益在前 250 步取得，两组独立实验给出一致结论，同等效果算力可减半。
3. **未观察到自我反思涌现** —— 反思词频全程在 0.6%–1.4% 之间。在 1.5B 规模、500 步预算下，没有复现 R1-Zero 论文中的涌现现象。这是一个负结果，如实记录。

---

## 方法论：这个项目真正想展示的东西

### 1. 训练 reward 上涨 ≠ 模型变强

Stage 1 的训练曲线非常好看：reward 0.543 → 0.654，训练集正确率 0.401 → 0.495。
但测试集上准确率的提升 *p*=0.322，而**格式合规率**的提升 *p*=1.2e-9。

涨的几乎全是格式，不是推理能力。

这个诊断能成立，依赖一个设计决定：**三个奖励函数分开注册**，而不是合成一个总分。

```python
def make_reward_funcs(cfg):
    def r_format(completions, **kw): ...
    def r_answer(completions, gold=None, **kw): ...
    def r_length(completions, **kw): ...
    return [r_format, r_answer, r_length]
```

TRL 会按函数名分别记录 `rewards/r_format/mean`、`rewards/r_answer/mean`。
如果合成一个总分，这个关键发现就看不见了。

### 2. 只有配对检验能识破假信号

两次评测跑的是**同一批题目**，属于配对数据。直接比两个准确率会浪费掉配对结构。

```python
def mcnemar(a_flags, b_flags):
    b01 = sum(1 for x, y in zip(a_flags, b_flags) if not x and y)   # 由错变对
    b10 = sum(1 for x, y in zip(a_flags, b_flags) if x and not y)   # 由对变错
    chi2 = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)                  # 连续性校正
    return erfc(sqrt(chi2 / 2))
```

Stage 1 的 52.2% → 55.0%，逐题看是 **93 道变对、79 道变错，净赚 14 道 / 500**。
McNemar 只看不一致对，统计效力远高于把两组当独立样本。

### 3. 从诊断到验证：单变量受控实验

Stage 1 效果不显著，可能是方法不行，也可能是训练太保守。
突破口是一个平时没人看的指标：**KL 全程 ≤ 0.001 —— 模型根本没有被推动。**

据此设计两组对照，**每组只改一个变量**，其余全部固定：

| 组 | 唯一改动 | 隔离的变量 |
|---|---|---|
| A1 | lr 1e-6 → 1e-5，200 → 500 步 | 学习率是否过小 |
| A2 | 关掉 LoRA 改全参，lr=2e-6，500 步 | LoRA 低秩约束是否限制容量 |

同一套代码、同一套奖励、同一批测试题，只改学习率和步数，
结果从「+2.8 点，*p*=0.32（噪声）」变成「**+19.4 点，*p*<1e-13**」。

诊断被证实。

### 4. 机制解释：熵曲线与准确率曲线的时间点吻合

```
A1 训练轨迹
  1–40 步    熵 0.670   ← 学习期
 81–120 步   熵 0.388
161–200 步   熵 0.288   ← 降到平台
241–280 步   熵 0.297   ← 之后一直横着
```

熵衡量输出多样性。**熵在 160–200 步降到平台，准确率提升也正好在那时停止。**
探索停止 → 学习停止。两条独立曲线的时间点吻合，构成完整的机制解释，
也解释了结论 2（后一半训练白跑）的成因。

### 5. 自定义诊断指标

TRL 自带的指标不够用，补了 GRPO 特有的两个：

```python
# 零方差组占比：一组 G 条回答全对或全错 → 组内 advantage 归零 → 这批采样白烧
groups = correct.reshape(-1, self.G)
zero_var = float(np.mean(groups.std(axis=1) < 1e-8))
```

**这里有一个定义差异值得注意**：TRL 自带的 `frac_reward_zero_std` 全程为 0，
因为它看的是**总奖励**的组内标准差（连续值，几乎不会恰好为 0）；
本仓库的 `diag/zero_var_group_ratio` 看的是**答案正确性**（二值，经常整组全对）。

衡量「采样是否被浪费」，后者才有意义。而这个指标在训练中**持续恶化：0.181 → 0.294**——
模型越强，一组回答全对的情况越多，advantage 归零，采样白烧。
目前已有约 29% 的采样被浪费，而采样占训练总时间的 60–80%。

这是下一阶段的题目，**动机来自本项目自己的实验数据，不是从论文里挑的**。

---

## 技术栈

**TRL**（GRPO）+ **PEFT/LoRA**（省显存）+ **vLLM**（加速采样）+ **transformers**

双卡分工：一张卡专职 vLLM 采样，另一张专职训练。

---

## 目录结构

```
grpo-math/
├── configs/
│   ├── base.yaml              全局：模型、数据、prompt、奖励权重、GPU 分工
│   ├── grpo_1.5b.yaml         Stage 1 训练超参
│   ├── grpo_a1_lr.yaml        A1 实验（lr=1e-5，500 步）
│   └── grpo_a2_full.yaml      A2 实验（全参微调，500 步）
├── src/
│   ├── cfg.py                 配置加载，支持 ${a.b} 引用
│   ├── data.py                GSM8K 加载 + prompt 构造 + 标准答案抽取
│   ├── rewards.py         ★★★ 奖励函数（项目核心，自带 10 条边界用例）
│   ├── eval.py            ★★  评测（vLLM 批量生成 + 打分）
│   ├── train_grpo.py      ★★★ 训练入口
│   └── compare.py         ★★  配对显著性检验（McNemar）
├── scripts/
│   ├── start_vllm.sh          拉起 vLLM 采样服务
│   ├── run_train.sh           训练
│   ├── run_eval.sh            评测
│   └── run_stage2a.sh         A1/A2 全流程串行驱动
├── docs/progress.md           阶段进度记录
└── results/                   全部实验结果
```

---

## 快速开始

```bash
pip install -r requirements.txt
```

奖励函数自带 10 条边界用例，可直接运行自测（覆盖逗号数字 `1,234`、浮点等价 `72.0`、
空 answer、标签顺序颠倒等情况）：

```bash
python src/rewards.py
```

> **奖励函数写完必须像单元测试一样验证。** 这是整个项目唯一注入先验的地方，
> 也最容易出 bug —— 写错了模型会学到奇怪的东西。

起 vLLM 采样服务：

```bash
bash scripts/start_vllm.sh
```

训练：

```bash
bash scripts/run_train.sh configs/grpo_a1_lr.yaml
```

评测（LoRA 与全参分别用 `--lora` / `--model`）：

```bash
bash scripts/run_eval.sh <tag> --lora <adapter路径>
```

任意两次评测之间做配对显著性检验：

```bash
python src/compare.py --before baseline_base_1.5b --after eval_s2a1_lora_lr1e-5
```

---

## 结果怎么看

| 想看什么 | 看哪里 |
|---|---|
| 汇总数字 | `results/<tag>_summary.json`（准确率、格式合规率、长度分位数、涌现词频、无法解析比例） |
| 逐题明细 | `results/<tag>.jsonl`（含模型完整输出，怀疑数字有问题时人工抽读） |
| 训练曲线 | `results/train_<run_name>.log`，每步一行 JSON；TensorBoard 在 `<output_dir>/runs/` |
| 训练过程快照 | `results/s1_smoke_1.5b/train_samples_step*.jsonl`，每 25 步存 16 条真实输出 |

**人工抽读逐题明细是必做动作。** Stage 0 就是靠它确认 52.2% 的基线不是解析脚本误判
——原本预估基线 10–30%（「基座模型不会自发推理」），实际 52.2%，
原因是 Qwen2.5 系列在预训练阶段就吃了大量数学数据，这个前提对它不成立。

---

## 全部实验结果

| tag | 准确率 | 格式合规 | 平均长度 | 反思词频 | 无法解析 |
|---|---|---|---|---|---|
| baseline_base_1.5b | 0.522 | 0.852 | 164.7 | 0.006 | 0.060 |
| s1_after_200steps | 0.550 | 0.966 | 163.9 | 0.010 | 0.024 |
| eval_a1_step250 | 0.704 | 0.996 | 178.8 | 0.012 | 0.010 |
| **eval_s2a1_lora_lr1e-5** | **0.716** | 0.998 | 171.2 | 0.008 | 0.000 |
| eval_a2_step250 | 0.712 | 0.998 | 196.4 | 0.014 | 0.002 |
| **eval_s2a2_full_lr2e-6** | **0.720** | 0.996 | 165.9 | 0.008 | 0.002 |

---

## 一个额外观察：全参与 LoRA 的风格差异

| | stepwise 词频 | conclude 词频 |
|---|---|---|
| 基线 | 0.520 | 0.268 |
| A1（LoRA） | 0.484 | 0.268 |
| A2（全参） | **0.280** | **0.438** |

全参微调把模型的表达风格改得更多，LoRA 更贴近基座原有风格，但两者准确率统计不可区分。
从实验侧印证了低秩约束的含义：**LoRA 动得更少，却够用。**

---

## 下一阶段

围绕「RL 训练稳定性与训推一致性」展开：

- **训推一致性测量** —— 采样引擎（vLLM）返回的 logprob 与训练侧重算的 logprob 存在差异，
  破坏 GRPO 的 on-policy 假设。量化这个 gap 随训练步数、序列长度、LoRA/全参的变化。
- **重要性采样修正** —— truncated importance sampling / off-policy masking 的单变量对照。
- **采样预算分配** —— 针对零方差组占比 0.18 → 0.29 的持续恶化。
- **熵坍缩抑制** —— 熵在 160 步降到平台后学习停止。

---

## 踩过的坑

| 问题 | 根因 | 解决 |
|---|---|---|
| `hf download` 401 | 新版 huggingface_hub 走 Xet CAS，镜像站不代理 | `HF_HUB_DISABLE_XET=1` |
| `GRPOConfig` 报 `max_prompt_length` 非法 | TRL 1.13 已移除该字段 | 删除 |
| `GRPOConfig` 报 `logging_dir` 非法 | GRPOConfig 并非完整继承 `TrainingArguments` | 删除，TB 默认写 `output_dir/runs/` |
| `TensorBoardCallback requires tensorboard` | vLLM 和 TRL 都不依赖它 | `pip install tensorboard` |

**方法论教训**：前三次是一个参数一个参数试错。第四次改成**一次性把 28 个参数
对着 `dataclasses.fields()` 的 189 个真实字段全量核验**，一次通过。
