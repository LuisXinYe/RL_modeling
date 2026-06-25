#!/usr/bin/env python3
"""Inference performance demo — reads config from runtime YAML.

Demonstrates standalone inference (serving) performance modeling using
generation_time() from llm_perf.inference.

Usage:
    python examples/demo_inference.py                          # default: Llama-3.1-8B
    python examples/demo_inference.py --config runtime/llama3_1_8b_8x_910c.yaml
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
from llm_perf.inference import generation_time, effective_response_len

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
        "DeepSeek-V4": "deepseekv4",
    }
    stem = mapping.get(model_name, model_name)
    return CONFIGS / "models" / f"{stem}.yaml"


def main():
    parser = argparse.ArgumentParser(description="Inference performance demo")
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

    # Use gen_parallelism if available, otherwise use parallelism
    par = rt.gen_parallelism or rt.parallelism

    # Build an WorkloadConfig from InferenceConfig for generation_time() compatibility
    inf = rt.inference
    rl_cfg = WorkloadConfig(
        group_size=1,  # standalone inference: one response per prompt
        avg_prompt_len=inf.avg_prompt_len,
        avg_response_len=inf.avg_response_len,
        max_response_len=inf.max_response_len,
        std_response_len=inf.std_response_len,
        gen_batch_size=inf.batch_size,
        use_speculative_decoding=inf.use_speculative_decoding,
        mtp_acceptance_len=inf.mtp_acceptance_len,
    )

    # Print config summary
    print(f"Model:    {model.name}")
    print(f"Hardware: {hw.name}")
    print(f"Network:  {rt.network or 'default'}")
    print(f"Parallelism: TP={par.tp} PP={par.pp} EP={par.ep} DP={par.dp}")
    print(f"Inference: batch_size={inf.batch_size}, prompt={inf.avg_prompt_len}, "
          f"response={inf.avg_response_len}, max_response={inf.max_response_len}")
    print()

    # Run inference timing
    prefill_sim, t_step = generation_time(model, hw, par, rl_cfg)

    # Compute throughput metrics
    eff_len = effective_response_len(
        avg=inf.avg_response_len,
        std=inf.std_response_len,
        batch_size=inf.batch_size,
        max_len=inf.max_response_len,
    )
    tps = inf.batch_size * eff_len / t_step if t_step > 0 else 0
    sps = inf.batch_size / t_step if t_step > 0 else 0

    print("=" * 60)
    print(" Inference Performance Report")
    print("=" * 60)
    print(f"  Total time:         {t_step:.3f} s")
    print(f"  Effective resp len: {eff_len:.0f} tokens")
    print(f"  Throughput:         {tps:,.0f} tokens/s")
    print(f"  Samples/s:          {sps:.2f}")
    print(f"  Prefill compute:    {prefill_sim.wall_clock_time:.3f} s")
    print(f"  Decode per token:   {(t_step - prefill_sim.wall_clock_time) / eff_len:.4f} s/token")
    print(f"  Weight per device:  {prefill_sim.weight_bytes / 1e9:.2f} GB")
    print("=" * 60)


if __name__ == "__main__":
    main()