#!/usr/bin/env python3
"""RL post-training performance demo — reads config from runtime YAML.

Usage:
    python examples/demo_rl.py                          # default: DeepSeek-V4
    python examples/demo_rl.py --config deepseekv4_256x_910c
"""

import argparse
from pathlib import Path

import yaml

from llm_perf.config import (
    load_model_config,
    load_hardware_config,
    load_runtime_config,
    WorkloadConfig,
    ParallelismConfig,
)
from llm_perf.model import LLMPerformanceModel
from llm_perf.report import format_table

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"


def resolve_hardware_path(hw_name: str) -> Path:
    """Resolve hardware name to YAML path."""
    mapping = {
        "Ascend 910B": "ascend_910b",
        "Ascend 910C": "ascend_910c",
        "CloudMatrix 384": "cloudmatrix_384",
    }
    stem = mapping.get(hw_name, hw_name)
    return CONFIGS / "hardware" / f"{stem}.yaml"


def resolve_model_path(model_name: str) -> Path:
    """Resolve model name to YAML path."""
    mapping = {
        "Llama-3.1-8B": "llama3_1_8b",
        "Qwen2.5-72B": "qwen2_5_72b",
        "Mistral-7B": "mistral_7b",
        "Qwen3-235B-A22B": "qwen3_235b_moe",
        "DeepSeek-V3-671B": "deepseekv3_671b",
        "DeepSeek-V4": "deepseekv4",
    }
    stem = mapping.get(model_name, model_name)
    return CONFIGS / "models" / f"{stem}.yaml"


def main():
    parser = argparse.ArgumentParser(description="RL post-training performance demo")
    parser.add_argument(
        "--config", default="deepseekv4_256x_910c",
        help="Runtime config stem (e.g. deepseekv4_256x_910c) or full path",
    )
    args = parser.parse_args()

    # Load runtime config
    rt_path = Path(args.config)
    if not rt_path.is_absolute() and not rt_path.exists():
        rt_path = CONFIGS / "runtime" / f"{rt_path}.yaml"
    rt = load_runtime_config(str(rt_path))

    # Resolve and load model + hardware
    model = load_model_config(str(resolve_model_path(rt.model)))
    hw = load_hardware_config(str(resolve_hardware_path(rt.hardware)))

    # Apply network topology overrides
    if rt.network:
        from llm_perf.config import HardwareConfig
        net_mapping = {
            "HCCS+RoCE (910B)": "hccs_roce_910b",
            "HCCS+RoCE (910C)": "hccs_roce_910c",
            "CloudMatrix 384 Fullmesh": "cloudmatrix_384_fullmesh",
        }
        net_stem = net_mapping.get(rt.network, rt.network)
        net_path = CONFIGS / "network" / f"{net_stem}.yaml"
        if net_path.exists():
            with open(net_path, encoding="utf-8") as f:
                net_data = yaml.safe_load(f)
            tiers = net_data.get("tiers", [])
            intra = next((t for t in tiers if t.get("name") == "intra_node"), tiers[0] if tiers else {})
            inter = next((t for t in tiers if t.get("name") == "inter_node"), tiers[-1] if len(tiers) > 1 else {})
            hw.intra_node_bw_gb_s = intra.get("bandwidth_gb_s", hw.intra_node_bw_gb_s)
            hw.intra_node_latency_us = intra.get("latency_us", hw.intra_node_latency_us)
            hw.inter_node_bw_gb_s = inter.get("bandwidth_gb_s", hw.inter_node_bw_gb_s)
            hw.inter_node_latency_us = inter.get("latency_us", hw.inter_node_latency_us)
            hw.devices_per_node = net_data.get("devices_per_node", hw.devices_per_node)

    perf = LLMPerformanceModel(model, hw)

    # Print config summary
    print(f"Model:    {model.name}")
    print(f"Hardware: {hw.name}")
    print(f"Network:  {rt.network or 'default'}")
    print(f"Devices:  {rt.total_devices}")
    print()

    # Parallelism configs
    train_par = rt.parallelism
    gen_par = rt.gen_parallelism or rt.parallelism
    ref_par = rt.ref_parallelism or rt.parallelism

    # RL config
    rl_cfg = rt.rl
    if rl_cfg is None:
        rl_cfg = WorkloadConfig()

    print(f"Generation parallelism: TP={gen_par.tp} PP={gen_par.pp} EP={gen_par.ep} DP={gen_par.dp}")
    print(f"Reference parallelism:  TP={ref_par.tp} PP={ref_par.pp} EP={ref_par.ep} DP={ref_par.dp}")
    print(f"Training parallelism:   TP={train_par.tp} PP={train_par.pp} EP={train_par.ep} DP={train_par.dp} CP={train_par.cp}")
    print(f"RL: group_size={rl_cfg.group_size}, gen_batch={rl_cfg.gen_batch_size}, train_batch={rl_cfg.train_batch_size}")
    print()

    # Run prediction
    print("=" * 60)
    print(" RL Post-Training Performance Report")
    print("=" * 60)
    report = perf.derive_targets(
        total_devices=rt.total_devices,
        rl_cfg=rl_cfg,
        gen_parallel=gen_par,
        train_parallel=train_par,
        ref_parallel=ref_par,
    )
    print(format_table(report))


if __name__ == "__main__":
    main()