"""
DPO 超参数消融实验
==================
面试亮点：展示你对 DPO 核心超参数的理解和实验能力

实验设计：
1. β (beta) 消融：控制偏好学习强度
2. Learning Rate 消融：DPO 对 lr 极其敏感
3. LoRA Rank 消融：参数效率与效果的权衡

用法：
    python dpo_ablation.py --exp beta --base_model merged-ecom-sft
    python dpo_ablation.py --exp lr --base_model merged-ecom-sft
    python dpo_ablation.py --exp all --base_model merged-ecom-sft
"""

import os
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional
import torch


@dataclass
class AblationConfig:
    """消融实验配置"""
    exp_name: str
    beta: float = 0.1
    learning_rate: float = 5e-7
    lora_rank: int = 8
    lora_alpha: int = 16
    max_steps: int = 150
    batch_size: int = 2
    grad_accum: int = 4
    
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum
    
    def to_cmd_args(self, base_model: str, data_dir: str, output_dir: str) -> list:
        """转换为命令行参数"""
        return [
            "python", "dpo_training.py",
            "--model_name_or_path", base_model,
            "--template_name", "qwen",
            "--train_file_dir", data_dir,
            "--validation_file_dir", data_dir,
            "--per_device_train_batch_size", str(self.batch_size),
            "--per_device_eval_batch_size", "1",
            "--do_train",
            "--do_eval",
            "--use_peft", "True",
            "--max_train_samples", "5000",
            "--max_eval_samples", "50",
            "--max_steps", str(self.max_steps),
            "--eval_steps", "30",
            "--save_steps", "50",
            "--max_source_length", "512",
            "--max_target_length", "1024",
            "--output_dir", output_dir,
            "--target_modules", "all",
            "--lora_rank", str(self.lora_rank),
            "--lora_alpha", str(self.lora_alpha),
            "--lora_dropout", "0.05",
            "--torch_dtype", "bfloat16",
            "--bf16", "True",
            "--fp16", "False",
            "--device_map", "auto",
            "--report_to", "tensorboard",
            "--remove_unused_columns", "False",
            "--gradient_checkpointing", "True",
            "--gradient_accumulation_steps", str(self.grad_accum),
            "--learning_rate", str(self.learning_rate),
            "--beta", str(self.beta),
        ]


# ═══════════════════════════════════════════════════════════════
# 消融实验配置
# ═══════════════════════════════════════════════════════════════

BETA_ABLATION = {
    "beta_0.01": AblationConfig(exp_name="beta_0.01", beta=0.01),   # 非常保守
    "beta_0.05": AblationConfig(exp_name="beta_0.05", beta=0.05),   # 保守
    "beta_0.1":  AblationConfig(exp_name="beta_0.1",  beta=0.1),    # 默认值
    "beta_0.2":  AblationConfig(exp_name="beta_0.2",  beta=0.2),    # 较激进
    "beta_0.5":  AblationConfig(exp_name="beta_0.5",  beta=0.5),    # 激进
}

LR_ABLATION = {
    "lr_1e-7": AblationConfig(exp_name="lr_1e-7", learning_rate=1e-7),
    "lr_5e-7": AblationConfig(exp_name="lr_5e-7", learning_rate=5e-7),   # 默认
    "lr_1e-6": AblationConfig(exp_name="lr_1e-6", learning_rate=1e-6),
    "lr_5e-6": AblationConfig(exp_name="lr_5e-6", learning_rate=5e-6),
    "lr_1e-5": AblationConfig(exp_name="lr_1e-5", learning_rate=1e-5),   # 高风险
}

RANK_ABLATION = {
    "rank_4":  AblationConfig(exp_name="rank_4",  lora_rank=4,  lora_alpha=8),
    "rank_8":  AblationConfig(exp_name="rank_8",  lora_rank=8,  lora_alpha=16),   # 默认
    "rank_16": AblationConfig(exp_name="rank_16", lora_rank=16, lora_alpha=32),
    "rank_32": AblationConfig(exp_name="rank_32", lora_rank=32, lora_alpha=64),
}

ALL_ABLATIONS = {
    "beta": BETA_ABLATION,
    "lr": LR_ABLATION,
    "rank": RANK_ABLATION,
}


def run_experiment(config: AblationConfig, base_model: str, data_dir: str, 
                   output_base: str, dry_run: bool = False) -> dict:
    """运行单个实验"""
    output_dir = os.path.join(output_base, config.exp_name)
    
    print(f"\n{'='*60}")
    print(f"  实验: {config.exp_name}")
    print(f"  配置: β={config.beta}, lr={config.learning_rate}, rank={config.lora_rank}")
    print(f"{'='*60}")
    
    cmd = config.to_cmd_args(base_model, data_dir, output_dir)
    
    if dry_run:
        print(f"  [DRY RUN] 命令: {' '.join(cmd[:10])}...")
        return {"status": "dry_run", "config": asdict(config)}
    
    start_time = datetime.now()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        success = result.returncode == 0
        
        # 解析训练日志中的关键指标
        metrics = parse_training_log(output_dir)
        
        return {
            "status": "success" if success else "failed",
            "config": asdict(config),
            "output_dir": output_dir,
            "duration_minutes": (datetime.now() - start_time).seconds / 60,
            "metrics": metrics,
            "stderr": result.stderr[-500:] if not success else "",
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "config": asdict(config)}
    except Exception as e:
        return {"status": "error", "config": asdict(config), "error": str(e)}


def parse_training_log(output_dir: str) -> dict:
    """解析训练日志获取关键指标"""
    metrics = {}
    trainer_state = os.path.join(output_dir, "trainer_state.json")
    
    if os.path.exists(trainer_state):
        with open(trainer_state) as f:
            state = json.load(f)
            
        # 提取关键指标
        log_history = state.get("log_history", [])
        
        train_losses = [l["loss"] for l in log_history if "loss" in l]
        eval_losses = [l["eval_loss"] for l in log_history if "eval_loss" in l]
        rewards_chosen = [l.get("rewards/chosen", 0) for l in log_history if "rewards/chosen" in l]
        rewards_rejected = [l.get("rewards/rejected", 0) for l in log_history if "rewards/rejected" in l]
        
        if train_losses:
            metrics["train_loss_final"] = train_losses[-1]
            metrics["train_loss_min"] = min(train_losses)
        if eval_losses:
            metrics["eval_loss_final"] = eval_losses[-1]
            metrics["eval_loss_min"] = min(eval_losses)
        if rewards_chosen and rewards_rejected:
            metrics["reward_margin_final"] = rewards_chosen[-1] - rewards_rejected[-1]
            metrics["reward_margin_max"] = max(c - r for c, r in zip(rewards_chosen, rewards_rejected))
            
    return metrics


def run_ablation_suite(exp_type: str, base_model: str, data_dir: str, 
                       output_base: str, dry_run: bool = False) -> dict:
    """运行一组消融实验"""
    if exp_type == "all":
        configs = {}
        for exp_name, exp_configs in ALL_ABLATIONS.items():
            configs.update(exp_configs)
    else:
        configs = ALL_ABLATIONS.get(exp_type, {})
    
    if not configs:
        raise ValueError(f"未知实验类型: {exp_type}")
    
    results = {}
    for name, config in configs.items():
        results[name] = run_experiment(config, base_model, data_dir, output_base, dry_run)
    
    return results


def generate_ablation_report(results: dict, output_path: str):
    """生成消融实验报告"""
    report = []
    report.append("# DPO 消融实验报告\n")
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # 汇总表格
    report.append("## 实验结果汇总\n\n")
    report.append("| 实验名称 | β | LR | Rank | Train Loss | Eval Loss | Reward Margin | 状态 |\n")
    report.append("|----------|------|------|------|------------|-----------|---------------|------|\n")
    
    for name, result in results.items():
        config = result.get("config", {})
        metrics = result.get("metrics", {})
        status = result.get("status", "unknown")
        
        train_loss = metrics.get("train_loss_final", "-")
        eval_loss = metrics.get("eval_loss_final", "-")
        reward_margin = metrics.get("reward_margin_final", "-")
        
        if isinstance(train_loss, float):
            train_loss = f"{train_loss:.4f}"
        if isinstance(eval_loss, float):
            eval_loss = f"{eval_loss:.4f}"
        if isinstance(reward_margin, float):
            reward_margin = f"{reward_margin:.3f}"
        
        report.append(f"| {name} | {config.get('beta', '-')} | {config.get('learning_rate', '-')} | "
                     f"{config.get('lora_rank', '-')} | {train_loss} | {eval_loss} | {reward_margin} | {status} |\n")
    
    report.append("\n## 关键发现\n\n")
    report.append("### β (Beta) 影响分析\n")
    report.append("- β 控制 DPO 损失中的 KL 散度惩罚强度\n")
    report.append("- 较小的 β (0.01-0.05): 更保守，不容易偏离 SFT 模型\n")
    report.append("- 较大的 β (0.2-0.5): 更激进地学习偏好，但可能过拟合\n\n")
    
    report.append("### Learning Rate 影响分析\n")
    report.append("- DPO 对学习率极其敏感\n")
    report.append("- 推荐范围: 1e-7 ~ 5e-6\n")
    report.append("- 过高的 lr (>1e-5) 容易导致训练不稳定\n\n")
    
    report.append("### LoRA Rank 影响分析\n")
    report.append("- Rank 越大，可学习参数越多\n")
    report.append("- 数据量小 (<1000条) 建议 rank=4~8\n")
    report.append("- 数据量大 (>5000条) 可用 rank=16~32\n\n")
    
    # 保存报告
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(report)
    
    print(f"\n✅ 报告已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="DPO 超参数消融实验")
    parser.add_argument("--exp", type=str, default="beta",
                       choices=["beta", "lr", "rank", "all"],
                       help="实验类型")
    parser.add_argument("--base_model", type=str, default="merged-ecom-sft",
                       help="SFT 后的基座模型路径")
    parser.add_argument("--data_dir", type=str, default="./data/reward",
                       help="DPO 数据目录")
    parser.add_argument("--output_base", type=str, default="./ablation_outputs",
                       help="实验输出目录")
    parser.add_argument("--dry_run", action="store_true",
                       help="仅打印命令，不执行")
    parser.add_argument("--report", type=str, default="ablation_report.md",
                       help="报告输出路径")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"  DPO 消融实验")
    print(f"  实验类型: {args.exp}")
    print(f"  基座模型: {args.base_model}")
    print(f"{'='*60}")
    
    # 检查 GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    
    # 运行实验
    results = run_ablation_suite(
        args.exp,
        args.base_model,
        args.data_dir,
        args.output_base,
        args.dry_run
    )
    
    # 保存结果
    os.makedirs(args.output_base, exist_ok=True)
    results_path = os.path.join(args.output_base, "ablation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 结果已保存: {results_path}")
    
    # 生成报告
    generate_ablation_report(results, args.report)


if __name__ == "__main__":
    main()
