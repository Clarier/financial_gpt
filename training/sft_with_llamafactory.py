# -*- coding: utf-8 -*-
"""
基于 LLaMA-Factory 的 SFT 训练脚本
=====================================
替换原有 supervised_finetuning.py，使用 LLaMA-Factory 框架进行监督微调。

LLaMA-Factory 优势：
  1. 内置 100+ 模型适配，开箱即用
  2. 原生支持多种训练方式（Full / LoRA / QLoRA / AQLM / GPTQ）
  3. 内置 FlashAttention-2、Unsloth 加速
  4. 支持多轮对话数据格式（sharegpt / alpaca）
  5. Web UI（LlamaBoard）可视化训练

适用模型：Qwen2.5 / LLaMA-3 / ChatGLM / Baichuan / DeepSeek 等

Usage:
  python sft_with_llamafactory.py \
      --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
      --dataset_path ./data/finetune_sharegpt/ecommerce_sft_sharegpt.jsonl \
      --output_dir outputs-ecom-sft \
      --num_train_epochs 3 \
      --use_lora True
"""

import os
import json
import yaml
import argparse
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# 1. 数据格式转换：转为 LLaMA-Factory 原生格式
# ══════════════════════════════════════════════════════════════════════

def convert_to_llamafactory_format(
    input_path: str,
    output_dir: str = "./data/llamafactory_sft",
    dataset_name: str = "ecommerce_sft",
    system_prompt: str = "你是专业的电商运营和销售顾问，拥有丰富的实战经验。",
) -> str:
    """
    将项目原始数据格式转换为 LLaMA-Factory sharegpt 格式。

    原始格式（alpaca-style）:
        {"instruction": "...", "input": "...", "output": "..."}

    LLaMA-Factory sharegpt 格式（支持多轮对话）:
        {"conversations": [
            {"from": "system", "value": "..."},
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "..."}
        ]}
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset_name}.json")

    converted = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)

            # 构建 conversations
            conversations = []

            # system prompt
            if system_prompt:
                conversations.append({
                    "from": "system",
                    "value": system_prompt
                })

            # user turn
            user_content = data.get("instruction", "")
            if data.get("input"):
                user_content += "\n" + data["input"]
            conversations.append({
                "from": "human",
                "value": user_content
            })

            # assistant turn
            conversations.append({
                "from": "gpt",
                "value": data.get("output", "")
            })

            converted.append({"conversations": conversations})

    # LLaMA-Factory 推荐 JSON 数组格式
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ 转换完成：{len(converted)} 条 → {output_path}")
    return output_path


def convert_multiturn_to_llamafactory(
    input_path: str,
    output_dir: str = "./data/llamafactory_sft",
    dataset_name: str = "ecommerce_multiturn",
    system_prompt: str = "你是专业的电商运营和销售顾问，拥有丰富的实战经验。",
) -> str:
    """
    将多轮对话数据转换为 LLaMA-Factory sharegpt 格式。

    输入格式（每行一个多轮对话）：
        {"conversations": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]}

    或者 messages 格式：
        {"messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]}
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset_name}.json")

    converted = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)

            conversations = []

            # 处理 messages 格式
            raw_turns = data.get("messages") or data.get("conversations", [])

            has_system = False
            for turn in raw_turns:
                role = turn.get("role") or turn.get("from", "")
                content = turn.get("content") or turn.get("value", "")

                if role in ("system",):
                    conversations.append({"from": "system", "value": content})
                    has_system = True
                elif role in ("user", "human"):
                    conversations.append({"from": "human", "value": content})
                elif role in ("assistant", "gpt"):
                    conversations.append({"from": "gpt", "value": content})

            # 如果没有 system prompt，插入默认的
            if not has_system and system_prompt:
                conversations.insert(0, {"from": "system", "value": system_prompt})

            if len(conversations) >= 2:  # 至少要有一轮 human + gpt
                converted.append({"conversations": conversations})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ 多轮对话转换完成：{len(converted)} 条 → {output_path}")
    return output_path


# ══════════════════════════════════════════════════════════════════════
# 2. 生成 LLaMA-Factory 数据集描述文件
# ══════════════════════════════════════════════════════════════════════

def generate_dataset_info(
    dataset_dir: str = "./data/llamafactory_sft",
    datasets: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """
    生成 LLaMA-Factory 所需的 dataset_info.json 配置文件。
    """
    if datasets is None:
        datasets = {}

    # 自动扫描目录下的 JSON 文件
    for f in Path(dataset_dir).glob("*.json"):
        name = f.stem
        if name not in datasets:
            datasets[name] = {
                "file_name": f.name,
                "formatting": "sharegpt",
                "columns": {
                    "messages": "conversations",
                },
                "tags": {
                    "role_tag": "from",
                    "content_tag": "value",
                    "user_tag": "human",
                    "assistant_tag": "gpt",
                    "system_tag": "system",
                }
            }

    info_path = os.path.join(dataset_dir, "dataset_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(datasets, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ dataset_info.json 已生成：{info_path}")
    logger.info(f"   包含数据集：{list(datasets.keys())}")
    return info_path


# ══════════════════════════════════════════════════════════════════════
# 3. 生成 LLaMA-Factory 训练配置 YAML
# ══════════════════════════════════════════════════════════════════════

def generate_training_yaml(
    model_name_or_path: str,
    dataset_dir: str,
    dataset_names: str,
    output_dir: str,
    config_path: str = "./train_sft.yaml",
    # 训练超参
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 2e-5,
    lr_scheduler_type: str = "cosine",
    warmup_ratio: float = 0.05,
    max_length: int = 2048,
    # LoRA 配置
    use_lora: bool = True,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target: str = "all",
    # QLoRA
    use_qlora: bool = False,
    quantization_bit: int = 4,
    # 训练选项
    bf16: bool = True,
    gradient_checkpointing: bool = True,
    flash_attn: str = "auto",
    # 评估
    val_size: float = 0.02,
    eval_steps: int = 100,
    save_steps: int = 200,
    logging_steps: int = 10,
    save_total_limit: int = 3,
    report_to: str = "tensorboard",
) -> str:
    """
    生成 LLaMA-Factory YAML 训练配置文件。
    """
    config = {
        # === 模型 ===
        "model_name_or_path": model_name_or_path,
        "trust_remote_code": True,

        # === 训练方式 ===
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora" if use_lora else "full",

        # === 数据 ===
        "dataset_dir": dataset_dir,
        "dataset": dataset_names,
        "template": "qwen",        # Qwen2.5 使用 qwen 模板
        "cutoff_len": max_length,
        "overwrite_cache": True,
        "preprocessing_num_workers": 8,

        # === 训练超参 ===
        "output_dir": output_dir,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "lr_scheduler_type": lr_scheduler_type,
        "warmup_ratio": warmup_ratio,
        "weight_decay": 0.05,
        "max_grad_norm": 1.0,

        # === 精度与优化 ===
        "bf16": bf16,
        "gradient_checkpointing": gradient_checkpointing,
        "flash_attn": flash_attn,
        "optim": "adamw_torch",

        # === 日志与保存 ===
        "logging_steps": logging_steps,
        "save_steps": save_steps,
        "save_total_limit": save_total_limit,
        "report_to": report_to,
        "overwrite_output_dir": True,

        # === 评估 ===
        "val_size": val_size,
        "eval_strategy": "steps",
        "eval_steps": eval_steps,
        "per_device_eval_batch_size": per_device_train_batch_size,
    }

    # LoRA 配置
    if use_lora:
        config.update({
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lora_target": lora_target,
        })

    # QLoRA 配置
    if use_qlora:
        config.update({
            "quantization_bit": quantization_bit,
            "quantization_method": "bitsandbytes",
        })

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    logger.info(f"✅ 训练配置已生成：{config_path}")
    return config_path


# ══════════════════════════════════════════════════════════════════════
# 4. 启动 LLaMA-Factory 训练
# ══════════════════════════════════════════════════════════════════════

def run_llamafactory_train(config_path: str) -> None:
    """
    调用 LLaMA-Factory CLI 启动训练。
    """
    cmd = f"llamafactory-cli train {config_path}"
    logger.info(f"🚀 启动 LLaMA-Factory 训练：{cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode == 0:
        logger.info("✅ LLaMA-Factory SFT 训练完成！")
    else:
        logger.error(f"❌ 训练失败，返回码：{result.returncode}")
        raise RuntimeError(f"Training failed with return code {result.returncode}")


def export_model(
    model_name_or_path: str,
    adapter_path: str,
    output_dir: str,
    export_config_path: str = "./export_config.yaml",
) -> None:
    """
    使用 LLaMA-Factory CLI 导出合并后的模型（替代 merge_peft_adapter.py）。
    """
    config = {
        "model_name_or_path": model_name_or_path,
        "adapter_name_or_path": adapter_path,
        "template": "qwen",
        "finetuning_type": "lora",
        "export_dir": output_dir,
        "export_size": 5,   # GB per shard
        "export_device": "cpu",
        "export_legacy_format": False,
    }

    with open(export_config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    cmd = f"llamafactory-cli export {export_config_path}"
    logger.info(f"🔧 导出合并模型：{cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode == 0:
        logger.info(f"✅ 模型已导出至：{output_dir}")
    else:
        logger.error(f"❌ 导出失败，返回码：{result.returncode}")


# ══════════════════════════════════════════════════════════════════════
# 5. 主函数
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="LLaMA-Factory SFT Training for E-commerce")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset_path", type=str, default="./data/finetune/ecommerce_sft.jsonl",
                        help="原始 SFT 数据路径（alpaca 格式 JSONL）")
    parser.add_argument("--multiturn_path", type=str, default=None,
                        help="多轮对话数据路径（可选）")
    parser.add_argument("--dataset_dir", type=str, default="./data/llamafactory_sft",
                        help="LLaMA-Factory 数据目录")
    parser.add_argument("--output_dir", type=str, default="outputs-ecom-sft")
    parser.add_argument("--merged_output_dir", type=str, default="merged-ecom-sft",
                        help="合并后模型输出路径")

    # 训练超参
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=2048)

    # LoRA
    parser.add_argument("--use_lora", type=bool, default=True)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--use_qlora", type=bool, default=False)

    # 控制
    parser.add_argument("--skip_convert", action="store_true", help="跳过数据转换")
    parser.add_argument("--skip_train", action="store_true", help="跳过训练（只做数据转换）")
    parser.add_argument("--skip_export", action="store_true", help="跳过模型导出")

    args = parser.parse_args()

    # Step 1: 数据转换
    dataset_names = []

    if not args.skip_convert:
        logger.info("=" * 60)
        logger.info("  Step 1: 数据格式转换")
        logger.info("=" * 60)

        # 转换单轮 SFT 数据
        if os.path.exists(args.dataset_path):
            convert_to_llamafactory_format(
                input_path=args.dataset_path,
                output_dir=args.dataset_dir,
                dataset_name="ecommerce_sft",
            )
            dataset_names.append("ecommerce_sft")

        # 转换多轮对话数据（如果有）
        if args.multiturn_path and os.path.exists(args.multiturn_path):
            convert_multiturn_to_llamafactory(
                input_path=args.multiturn_path,
                output_dir=args.dataset_dir,
                dataset_name="ecommerce_multiturn",
            )
            dataset_names.append("ecommerce_multiturn")

        # 生成 dataset_info.json
        generate_dataset_info(dataset_dir=args.dataset_dir)
    else:
        # 自动检测已有数据集
        for f in Path(args.dataset_dir).glob("*.json"):
            if f.stem != "dataset_info":
                dataset_names.append(f.stem)

    if not dataset_names:
        logger.error("未找到任何数据集！请检查数据路径。")
        return

    # Step 2: 生成训练配置
    logger.info("=" * 60)
    logger.info("  Step 2: 生成训练配置")
    logger.info("=" * 60)

    config_path = generate_training_yaml(
        model_name_or_path=args.model_name_or_path,
        dataset_dir=args.dataset_dir,
        dataset_names=",".join(dataset_names),
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        use_qlora=args.use_qlora,
    )

    # Step 3: 启动训练
    if not args.skip_train:
        logger.info("=" * 60)
        logger.info("  Step 3: 启动 LLaMA-Factory 训练")
        logger.info("=" * 60)
        run_llamafactory_train(config_path)

    # Step 4: 导出合并模型
    if not args.skip_export:
        logger.info("=" * 60)
        logger.info("  Step 4: 导出合并模型")
        logger.info("=" * 60)
        export_model(
            model_name_or_path=args.model_name_or_path,
            adapter_path=args.output_dir,
            output_dir=args.merged_output_dir,
        )

    logger.info("🎉 SFT Pipeline 全部完成！")


if __name__ == "__main__":
    main()