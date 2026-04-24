"""
基于论文的数据多样性增强模块
=====================================
本模块的每一项技术均有明确论文来源，不依赖人工预设列表。

论文依据：
┌─────────────────────────────────────────────────────────────────────┐
│ 技术 1: Evol-Instruct（In-Depth + In-Breadth 进化）                  │
│   WizardLM: Empowering LLMs to Follow Complex Instructions           │
│   Xu et al., ICLR 2024 | arXiv:2304.12244                           │
│   核心：从已有指令出发，用6种操作生成更复杂/更多样的新指令              │
│                                                                       │
│ 技术 2: Self-Instruct（种子自举）                                     │
│   Self-Instruct: Aligning LMs with Self-Generated Instructions       │
│   Wang et al., ACL 2023 | arXiv:2212.10560                          │
│   核心：从少量种子指令引导模型持续生成新指令，种子越多样生成越多样       │
│                                                                       │
│ 技术 3: D3方法（多样性 + 难度 + 可信度三维打分）                       │
│   D3: Diversity, Difficulty, Dependability-Aware Data Selection      │
│   Zhang et al., IJCAI 2025 | arXiv:2503.11441                       │
│   核心：用sample distinctiveness度量多样性，避免冗余                   │
│   特别：论文在淘宝直播场景实验，与本项目电商场景高度对齐               │
│                                                                       │
│ 技术 4: NovelSum多样性度量                                            │
│   Measuring Data Diversity for Instruction Tuning                    │
│   Semantic Scholar 2024                                               │
│   核心：sample-level novelty，与模型性能相关性达0.97                  │
│                                                                       │
│ 技术 5: ROUGE-L过滤（Self-Instruct原版方法）                          │
│   Wang et al., 2023：用ROUGE-L相似度>0.7过滤近似重复指令              │
└─────────────────────────────────────────────────────────────────────┘

用法：
    python evol_diversity.py \\
        --input data/finetune/ecommerce_sft.jsonl \\
        --output data/finetune/ecommerce_sft_evolved.jsonl \\
        --model qwen2.5:7b \\
        --evolve_rounds 3 \\
        --target_size 800
"""

import json
import re
import random
import time
import hashlib
import math
import argparse
import sys
from pathlib import Path
from typing import Optional
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════
# Ollama 封装（复用原项目结构）
# ══════════════════════════════════════════════════════════════════════

def _get_ollama():
    try:
        import ollama
        return ollama
    except ImportError:
        sys.exit("❌ 请先安装: pip install ollama tqdm")


class LLMBackend:
    def __init__(self, model: str = "qwen2.5:7b"):
        self.ollama = _get_ollama()
        self.model = model
        print(f"✅ 模型: {model}")

    def chat(self, system: str, user: str,
             max_tokens: int = 1500, temperature: float = 0.85) -> str:
        try:
            resp = self.ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                options={
                    "num_predict":    max_tokens,
                    "temperature":    temperature,
                    "top_p":          0.9,
                    "repeat_penalty": 1.1,
                },
            )
            text = resp["message"]["content"]
            for tok in ["<|im_end|>", "<|im_start|>", "<|endoftext|>", "assistant\n"]:
                text = text.replace(tok, "")
            return re.sub(r"\n{3,}", "\n\n", text).strip()
        except Exception as e:
            return ""

    def chat_retry(self, system: str, user: str,
                   max_tokens: int = 1500, temperature: float = 0.85,
                   retries: int = 3) -> str:
        for attempt in range(retries):
            result = self.chat(system, user, max_tokens, temperature)
            if result:
                return result
            time.sleep(2 ** attempt)
        return ""


# ══════════════════════════════════════════════════════════════════════
# 技术1+2：Evol-Instruct（WizardLM, ICLR 2024）
# ══════════════════════════════════════════════════════════════════════

class EvolInstruct:
    """
    实现 WizardLM Evol-Instruct 的六种操作：
    
    In-Depth（深化）—— 5种，使指令更复杂：
      1. add_constraints   : 增加约束/条件
      2. deepening         : 加深问题深度
      3. concretize        : 具体化，增加具体场景细节
      4. increase_steps    : 增加推理步骤数
      5. complicate_input  : 复杂化输入条件

    In-Breadth（拓宽）—— 1种，增加指令多样性（类似Self-Instruct种子扩展）：
      6. breadth_mutation  : 基于原指令创造一个同领域但更罕见的新指令

    论文原文（WizardLM, arXiv:2304.12244）核心prompt摘录：
      In-Depth: "rewrite a given prompt into a more complex version"
      In-Breadth: "draw inspiration from the #Given Prompt# to create a brand new prompt
                   ...should belong to the same domain but be even more rare"
    """

    # ── 6个进化操作的prompt（严格对应论文描述）──
    EVOL_PROMPTS = {
        # In-Depth 1：增加约束
        "add_constraints": """\
你是一位电商运营专家。请将以下电商场景指令改写为更复杂的版本。
改写方式：在原指令的基础上，增加2~3个额外的约束条件或限制要求，使任务更具挑战性。
要求：
- 新指令必须合理、可被人类理解并作答
- 保持原指令的核心场景和意图
- 不要在新指令中说明你做了什么改动
- 只输出改写后的新指令，不要任何解释

原指令：{instruction}

新指令：""",

        # In-Depth 2：加深
        "deepening": """\
你是一位电商运营专家。请将以下电商场景指令改写为更复杂的版本。
改写方式：加深问题的深度——要求作答者提供更底层的原理分析、数据支撑或行业洞察，而非表面操作建议。
要求：
- 新指令必须合理、可被人类理解并作答
- 保持原指令的核心场景
- 只输出改写后的新指令，不要任何解释

原指令：{instruction}

新指令：""",

        # In-Depth 3：具体化
        "concretize": """\
你是一位电商运营专家。请将以下电商场景指令改写为更复杂的版本。
改写方式：具体化——为指令添加更详细的背景信息，如具体品类、平台、数据指标、用户群体、时间节点等。
要求：
- 新指令必须合理、可被人类理解并作答
- 保持原指令的核心场景
- 只输出改写后的新指令，不要任何解释

原指令：{instruction}

新指令：""",

        # In-Depth 4：增加推理步骤
        "increase_steps": """\
你是一位电商运营专家。请将以下电商场景指令改写为更复杂的版本。
改写方式：要求作答者按照多个明确的步骤或阶段来回答，增加推理链的长度（例如要求分析→诊断→方案→执行→评估）。
要求：
- 新指令必须合理、可被人类理解并作答
- 保持原指令的核心场景
- 只输出改写后的新指令，不要任何解释

原指令：{instruction}

新指令：""",

        # In-Depth 5：复杂化输入
        "complicate_input": """\
你是一位电商运营专家。请将以下电商场景指令改写为更复杂的版本。
改写方式：复杂化输入条件——引入多个相互制约的因素（如预算有限、时间紧迫、竞争激烈、团队人少等），
使问题变成一个需要权衡多方因素的复合决策题。
要求：
- 新指令必须合理、可被人类理解并作答
- 保持原指令的核心场景
- 只输出改写后的新指令，不要任何解释

原指令：{instruction}

新指令：""",

        # In-Breadth：拓宽（等同Self-Instruct的种子扩展）
        "breadth_mutation": """\
你是一位资深电商运营专家。请以下面这个电商场景指令为灵感，创造一个全新的电商指令。

要求（来自WizardLM论文的In-Breadth Evolving）：
- 新指令必须属于电商营销/运营/销售领域
- 新指令要比原指令更少见、更小众，覆盖原指令未涉及的细分场景
- 新指令的长度和难度与原指令相近
- 新指令必须合理，可被人类理解并作答
- 不要在新指令中提及"原指令"或"灵感来源"
- 只输出新指令本身，不要任何解释

原指令（灵感来源）：{instruction}

新指令：""",
    }

    def __init__(self, llm: LLMBackend):
        self.llm = llm
        # 论文比例：5种in-depth操作 + 1种in-breadth，等概率采样
        self.operations = list(self.EVOL_PROMPTS.keys())
        self.stats = {op: {"tried": 0, "success": 0} for op in self.operations}

    def evolve_one(self, instruction: str) -> tuple[str, str]:
        """
        对一条指令执行一次随机进化操作
        返回 (新指令, 使用的操作名)
        """
        op = random.choice(self.operations)
        prompt = self.EVOL_PROMPTS[op].replace("{instruction}", instruction)
        self.stats[op]["tried"] += 1

        new_instruction = self.llm.chat_retry(
            system="你是一位专业的电商指令改写专家，严格按照要求输出。",
            user=prompt,
            max_tokens=300,
            temperature=0.85,
        )
        if new_instruction and self._is_valid_evolution(instruction, new_instruction):
            self.stats[op]["success"] += 1
            return new_instruction.strip(), op
        return "", op

    def _is_valid_evolution(self, original: str, evolved: str) -> bool:
        """
        进化失败的4种判断（对应论文Elimination Evolving）：
        1. 进化后指令太短（信息量损失）
        2. 进化后与原指令完全相同
        3. 包含"原指令"等元提示词泄漏
        4. 长度超过原指令5倍（偏离过远）
        """
        if len(evolved) < 15:
            return False
        if evolved.strip() == original.strip():
            return False
        meta_words = ["原指令", "灵感来源", "改写前", "给定指令", "以下指令"]
        if any(w in evolved for w in meta_words):
            return False
        if len(evolved) > len(original) * 5:
            return False
        return True

    def evolve_dataset(self, instructions: list[str],
                       rounds: int = 2,
                       show_progress: bool = True) -> list[dict]:
        """
        对种子指令集执行多轮进化（Self-Instruct的bootstrapping思想）
        每轮：对当前所有指令各进化一次，成功的加入下一轮的种子池

        论文参数：WizardLM 执行 M=4 轮进化
        """
        all_evolved = []
        current_pool = list(instructions)

        for round_idx in range(1, rounds + 1):
            print(f"\n  ── 进化轮次 {round_idx}/{rounds}，当前种子池: {len(current_pool)} 条 ──")
            new_this_round = []

            iter_ = tqdm(current_pool, desc=f"  Round {round_idx}") if show_progress else current_pool
            for instr in iter_:
                new_instr, op = self.evolve_one(instr)
                if new_instr:
                    new_this_round.append({
                        "instruction": new_instr,
                        "source_instruction": instr,
                        "evol_operation": op,
                        "evol_round": round_idx,
                    })

            all_evolved.extend(new_this_round)
            # 成功进化的指令加入下一轮种子池（cumulative pool）
            current_pool.extend([d["instruction"] for d in new_this_round])
            print(f"  本轮成功: {len(new_this_round)} 条，累计进化: {len(all_evolved)} 条")

        return all_evolved

    def print_stats(self):
        print("\n  📊 各操作成功率：")
        for op, s in self.stats.items():
            rate = s["success"] / max(s["tried"], 1) * 100
            print(f"    {op:<25} {s['success']:>3}/{s['tried']:<3} = {rate:.0f}%")


# ══════════════════════════════════════════════════════════════════════
# 技术3+4：多样性度量（D3 + NovelSum 思想）
# ══════════════════════════════════════════════════════════════════════

class DiversityMeasurer:
    """
    基于 D3（IJCAI 2025）和 NovelSum（Semantic Scholar 2024）的多样性度量

    D3论文中的sample distinctiveness定义：
        一个样本的多样性分 = 它与其最近邻样本的距离
        （距离越大 = 越独特 = 多样性贡献越大）

    NovelSum论文核心：
        diversity = Σ novelty(sample_i)
        novelty(sample_i) = 1 - max_similarity(sample_i, 已选集合)
        与模型性能相关性达0.97

    本实现使用字符级n-gram特征的Jaccard距离作为计算高效的代理指标
    （不需要embedding模型，适合本地Ollama环境）
    """

    def __init__(self, ngram: int = 3):
        self.ngram = ngram

    def _get_ngrams(self, text: str) -> set:
        text = re.sub(r'\s+', ' ', text.lower().strip())
        return {text[i:i+self.ngram] for i in range(len(text) - self.ngram + 1)}

    def _jaccard_similarity(self, a: str, b: str) -> float:
        sa, sb = self._get_ngrams(a), self._get_ngrams(b)
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def novelty_score(self, candidate: str, existing: list[str]) -> float:
        """
        NovelSum论文：novelty = 1 - 与已有集合中最相似样本的相似度
        越高 = 越新颖 = 对多样性贡献越大
        """
        if not existing:
            return 1.0
        max_sim = max(self._jaccard_similarity(candidate, e) for e in existing[-200:])
        return 1.0 - max_sim

    def rouge_l_similarity(self, a: str, b: str) -> float:
        """
        Self-Instruct（Wang et al., 2023）原版过滤方法：
        ROUGE-L > 0.7 视为近似重复，应过滤
        这里用LCS近似实现
        """
        words_a = a.split()
        words_b = b.split()
        if not words_a or not words_b:
            return 0.0
        # LCS长度（DP）
        m, n = len(words_a), len(words_b)
        # 为效率限制长度
        words_a = words_a[:50]
        words_b = words_b[:50]
        m, n = len(words_a), len(words_b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if words_a[i-1] == words_b[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        lcs = dp[m][n]
        precision = lcs / n if n else 0
        recall = lcs / m if m else 0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def select_diverse_subset(self, candidates: list[str],
                               budget: int,
                               rouge_threshold: float = 0.7) -> list[str]:
        """
        贪心多样性选择（D3论文的subset selection思想）：
        1. 先用ROUGE-L过滤近似重复（Self-Instruct方法）
        2. 用novelty score贪心选取多样性最高的样本

        论文结论（D3, IJCAI 2025）：
        使用不到10%的数据，通过D3选择可达到全量数据训练的效果
        """
        if not candidates:
            return []

        # Step 1: ROUGE-L去重（Self-Instruct原版方法）
        deduplicated = []
        for cand in candidates:
            too_similar = False
            for existing in deduplicated[-100:]:
                if self.rouge_l_similarity(cand, existing) > rouge_threshold:
                    too_similar = True
                    break
            if not too_similar:
                deduplicated.append(cand)

        # Step 2: 贪心novelty选择
        selected = []
        random.shuffle(deduplicated)
        for cand in deduplicated:
            if len(selected) >= budget:
                break
            nov = self.novelty_score(cand, selected)
            if nov > 0.3:  # 最低新颖度阈值
                selected.append(cand)
        return selected

    def compute_dataset_novelsum(self, instructions: list[str]) -> float:
        """
        计算整个数据集的NovelSum分数
        NovelSum = Σ novelty(i) / N
        论文声称与模型性能相关性0.97
        """
        if not instructions:
            return 0.0
        scores = []
        seen = []
        for instr in instructions:
            scores.append(self.novelty_score(instr, seen))
            seen.append(instr)
        return sum(scores) / len(scores)


# ══════════════════════════════════════════════════════════════════════
# 主流水线：生成回复 + 组装完整数据集
# ══════════════════════════════════════════════════════════════════════

RESPONSE_SYSTEM = """\
你是资深电商运营专家和销售顾问，精通淘宝、京东、拼多多、抖音等平台规则与营销技巧。
请对以下电商场景问题给出专业、实用、有干货的回答。

回答要求：
1. 给出具体可执行的方案、话术示例或操作步骤
2. 结合电商平台规则和消费者心理
3. 有数据支撑或案例参考
4. 逻辑清晰，有层次感（可用数字编号）
5. 字数 400~700字，不要废话
"""


def generate_response(llm: LLMBackend, instruction: str) -> str:
    return llm.chat_retry(
        system=RESPONSE_SYSTEM,
        user=instruction,
        max_tokens=1000,
        temperature=0.7,
    )


def run_pipeline(args):
    print(f"\n{'━'*60}")
    print(f"  🧬 Evol-Instruct 数据多样性增强流水线")
    print(f"  论文来源: WizardLM(ICLR 2024) + Self-Instruct(ACL 2023)")
    print(f"           + D3(IJCAI 2025) + NovelSum(2024)")
    print(f"{'━'*60}")

    # 1. 加载原始数据（提取指令作为种子）
    print(f"\n[1/5] 加载种子数据: {args.input}")
    seed_instructions = []
    if Path(args.input).exists():
        with open(args.input, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    instr = d.get("instruction", "")
                    if len(instr) >= 10:
                        seed_instructions.append(instr)
                except Exception:
                    continue
    else:
        # 无原始数据时使用内置少量电商种子（对应Self-Instruct的175个seed tasks）
        seed_instructions = [
            "如何为一款新上线的智能手表撰写高转化率的淘宝主图文案？",
            "店铺差评率突然升高，应该从哪些维度分析原因？",
            "直播带货时买家说价格太贵不想买，客服怎么回复？",
            "大促前两周应该做哪些准备来提升活动期间的GMV？",
            "私域用户沉默了3个月，怎么设计唤醒活动？",
        ]
        print(f"  ⚠️  未找到输入文件，使用内置{len(seed_instructions)}条种子指令")

    print(f"  种子指令数: {len(seed_instructions)}")

    # 2. 计算初始多样性（NovelSum基准）
    measurer = DiversityMeasurer()
    baseline_novelsum = measurer.compute_dataset_novelsum(seed_instructions)
    print(f"  初始 NovelSum 多样性分数: {baseline_novelsum:.4f}")

    # 3. Evol-Instruct 进化（WizardLM方法）
    print(f"\n[2/5] Evol-Instruct 进化（轮次={args.evolve_rounds}）")
    llm = LLMBackend(model=args.model)
    evolver = EvolInstruct(llm)
    evolved_records = evolver.evolve_dataset(
        seed_instructions,
        rounds=args.evolve_rounds,
    )
    evolver.print_stats()

    # 4. 多样性过滤（D3 + NovelSum + ROUGE-L）
    all_instructions = seed_instructions + [r["instruction"] for r in evolved_records]
    print(f"\n[3/5] 多样性过滤（ROUGE-L去重 + NovelSum贪心选择）")
    print(f"  候选总量: {len(all_instructions)}")
    target = min(args.target_size, len(all_instructions))
    selected = measurer.select_diverse_subset(all_instructions, budget=target)
    print(f"  选择后: {len(selected)} 条")

    evolved_novelsum = measurer.compute_dataset_novelsum(selected)
    print(f"  进化后 NovelSum 多样性分数: {evolved_novelsum:.4f}")
    print(f"  多样性提升: +{(evolved_novelsum - baseline_novelsum):.4f} "
          f"({(evolved_novelsum/max(baseline_novelsum,0.001)-1)*100:.1f}%)")

    # 5. 生成回复，组装完整SFT数据
    print(f"\n[4/5] 为进化指令生成高质量回复...")
    # 找出哪些是纯进化出来的新指令（原种子已有回复则跳过）
    seed_set = set(seed_instructions)

    # 加载原始数据的已有回复
    existing_responses = {}
    if Path(args.input).exists():
        with open(args.input, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    existing_responses[d.get("instruction", "")] = d.get("output", "")
                except Exception:
                    continue

    final_records = []
    new_count = 0
    with tqdm(total=len(selected), desc="  生成回复") as pbar:
        for instr in selected:
            if instr in existing_responses:
                # 原有数据直接复用
                final_records.append({
                    "instruction": instr,
                    "input": "",
                    "output": existing_responses[instr],
                    "source": "original",
                    "evol_operation": "none",
                })
            else:
                # 进化出的新指令需要生成回复
                response = generate_response(llm, instr)
                if response and len(response) >= 100:
                    # 找到对应的进化操作
                    op = "breadth_mutation"
                    for rec in evolved_records:
                        if rec["instruction"] == instr:
                            op = rec["evol_operation"]
                            break
                    final_records.append({
                        "instruction": instr,
                        "input": "",
                        "output": response,
                        "source": "evol_instruct",
                        "evol_operation": op,
                    })
                    new_count += 1
            pbar.update(1)

    # 6. 保存
    print(f"\n[5/5] 保存结果...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in final_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 打印最终报告
    op_dist = {}
    for r in final_records:
        op = r.get("evol_operation", "none")
        op_dist[op] = op_dist.get(op, 0) + 1

    print(f"\n{'━'*60}")
    print(f"  📊 最终报告")
    print(f"{'━'*60}")
    print(f"  输入种子数:     {len(seed_instructions)}")
    print(f"  进化生成数:     {len(evolved_records)}")
    print(f"  多样性过滤后:   {len(selected)}")
    print(f"  最终数据集:     {len(final_records)} 条")
    print(f"  其中新增回复:   {new_count} 条")
    print(f"\n  进化操作分布:")
    for op, cnt in sorted(op_dist.items(), key=lambda x: -x[1]):
        print(f"    {op:<25} {cnt} 条")
    print(f"\n  多样性度量（NovelSum）:")
    print(f"    进化前: {baseline_novelsum:.4f}")
    print(f"    进化后: {evolved_novelsum:.4f}")
    gain = (evolved_novelsum / max(baseline_novelsum, 0.001) - 1) * 100
    print(f"    提升:   +{gain:.1f}%")
    print(f"\n  输出路径: {args.output}")
    print(f"{'━'*60}\n")

    return final_records


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="基于WizardLM Evol-Instruct + D3的数据多样性增强"
    )
    parser.add_argument("--input",  type=str, default="data/finetune/ecommerce_sft.jsonl",
                        help="原始SFT数据路径（.jsonl）")
    parser.add_argument("--output", type=str, default="data/finetune/ecommerce_sft_evolved.jsonl",
                        help="输出路径")
    parser.add_argument("--model",  type=str, default="qwen2.5:7b",
                        help="Ollama模型名称")
    parser.add_argument("--evolve_rounds", type=int, default=2,
                        help="进化轮次（WizardLM论文建议M=4，本地资源有限建议2~3）")
    parser.add_argument("--target_size",   type=int, default=800,
                        help="目标数据集大小（经多样性过滤后保留）")
    parser.add_argument("--rouge_threshold", type=float, default=0.7,
                        help="ROUGE-L相似度过滤阈值（Self-Instruct原版：0.7）")
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
