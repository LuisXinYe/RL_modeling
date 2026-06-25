#!/usr/bin/env python3
"""Pretraining performance demo — reads config from runtime YAML.

Demonstrates standalone pretraining performance modeling using
pretraining_time() from llm_perf.training.

Usage:
    python examples/demo_pretraining.py                          # default: Llama-3.1-8B
    python examples/demo_pretraining.py --config runtime/llama3_1_8b_8x_910c.yaml
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
from llm_perf.training import pretraining_time

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"


def resolve_hardware_path(hw_name: str) -> Path:
    mapping = {
        "Ascend 910B": "ascend_910b",
        "Ascend 910C": "ascend_910c",
        "CloudMatrix 384": "cloudmatrix_384",
    }
    stem = mapping.get(hw_name, hw_name)
    return CONFIGS / "hardware" / f"{stem}.yaml"


def resolve_model_path(model_name: str) -> Path:
    mapping = {
        "Llama-3.1-8B": "llama3_1_8b",
        "Qwen2.5-72B": "qwen2_5_72b",
        "Mistral-7B": "mistral_7b",
        "Qwen3-235B-A22B": "qwen3_235b_moe",
        "DeepSeek-V3-671B": "deepseekv3_671b",
    }
    stem = mapping.get(model_name, model_name)
    return CONFIGS / "models" / f"{stem}.yaml"


def main():
    parser = argparse.ArgumentParser(description="Pretraining performance demo")
    parser.add_argument(
        "--config", default="llama3_1_8b_8x_910c",
        help="Runtime config stem (e.g. llama3_1_8b_8x_910c) or full path",
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

    par = rt.parallelism

    # Build an WorkloadConfig from TrainConfig for pretraining_time() compatibility.
    # pretraining_time() uses: train_batch_size, train_micro_batch_size,
    # gradient_accumulation_steps, avg_prompt_len + avg_response_len.
    # For pretraining, avg_prompt_len=0 and avg_response_len=avg_seq_len
    # since there are no prompts — only continuation tokens.
    tc = rt.train
    rl_cfg = WorkloadConfig(
        avg_prompt_len=0,
        avg_response_len=tc.avg_seq_len,
        train_micro_batch_size=tc.train_micro_batch_size,
        train_batch_size=tc.train_batch_size,
        gradient_accumulation_steps=tc.gradient_accumulation_steps,
    )

    # Print config summary
    print(f"Model:    {model.name}")
    print(f"Hardware: {hw.name}")
    print(f"Network:  {rt.network or 'default'}")
    print(f"Parallelism: TP={par.tp} PP={par.pp} EP={par.ep} DP={par.dp} CP={par.cp}")
    print(f"Train: avg_seq_len={tc.avg_seq_len}, micro_bs={tc.train_micro_batch_size}, "
          f"global_bs={tc.train_batch_size}, grad_acc={tc.gradient_accumulation_steps}")
    print()

    # Run pretraining timing
    t_step, train_sim, step_bd = pretraining_time(model, hw, par, rl_cfg)

    # Compute throughput metrics
    seq_len = tc.avg_seq_len
    tps = tc.train_batch_size * seq_len / t_step if t_step > 0 else 0

    print("=" * 60)
    print(" Pretraining Performance Report")
    print("=" * 60)
    print(f"  Step time:          {t_step:.3f} s")
    print(f"  Throughput:         {tps:,.0f} tokens/s")
    print(f"  Compute time:       {train_sim.compute_time:.3f} s")
    print(f"  Weight per device:  {train_sim.weight_bytes / 1e9:.2f} GB")
    print(f"  Peak activation:    {train_sim.peak_activation_bytes / 1e9:.2f} GB")
    print()
    print("  Breakdown:")
    sub_steps = [
        ("policy_update", step_bd.policy_update),
        ("pp_p2p", step_bd.pp_p2p),
        ("pp_bubble", step_bd.pp_bubble),
        ("recompute", step_bd.recompute),
        ("optim_offload", step_bd.optim_offload),
    ]
    for label, t in sub_steps:
        if t > 0:
            pct = t / step_bd.total * 100
            print(f"    {label:16s} {t:8.3f} s  ({pct:5.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()