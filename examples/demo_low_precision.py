#!/usr/bin/env python3
"""llm-perf low-precision training demo: compare bf16 vs fp8 vs fp4 precision recipes.

Demonstrates compare_precision() for Llama-3.1-8B on Ascend 910C hardware.
Shows per-recipe step time, speedup, communication reduction, and memory.
"""

from pathlib import Path

from llm_perf.config import (
    ParallelismConfig,
    WorkloadConfig,
    load_hardware_config,
    load_model_config,
)
from llm_perf.model import compare_precision
from llm_perf.precision import PrecisionConfig, TensorPrecision

ROOT = Path(__file__).resolve().parent.parent

# Load model and hardware configs
model = load_model_config(ROOT / "configs" / "models" / "llama3_1_8b.yaml")
hw = load_hardware_config(ROOT / "configs" / "hardware" / "ascend_910c.yaml")

# Parallelism: 4-way DP, no TP (simple config to highlight comm reduction)
parallel_cfg = ParallelismConfig(tp=1, dp=4)

# Workload: small training step for quick demo
rl_cfg = WorkloadConfig(
    group_size=4,
    avg_prompt_len=512,
    avg_response_len=512,
    train_micro_batch_size=2,
    train_batch_size=8,
    gen_batch_size=8,
)

# Precision recipes to compare
recipes = {
    "bf16": PrecisionConfig.bf16_default(),
    "fp8": PrecisionConfig(
        weights=TensorPrecision(dtype="fp8_e4m3", block_size=128),
        activations=TensorPrecision(dtype="fp8_e4m3", block_size=128),
        comm=TensorPrecision(dtype="fp8_e4m3"),
    ),
    "fp4": PrecisionConfig(
        weights=TensorPrecision(dtype="fp4_e2m1", block_size=16, scale_bytes=1),
        activations=TensorPrecision(dtype="fp4_e2m1", block_size=16, scale_bytes=1),
        comm=TensorPrecision(dtype="fp4_e2m1"),
    ),
}

print(f"Model:    {model.name}")
print(f"Hardware: {hw.name}")
print(f"  bf16 TFLOPS: {hw.peak_tflops_bf16:.0f}")
print(f"  fp8  TFLOPS: {hw.peak_tflops.get('fp8', 'N/A')}")
print(f"  fp4  TFLOPS: {hw.peak_tflops.get('fp4', 'N/A')}")
print(f"Parallelism: TP={parallel_cfg.tp} DP={parallel_cfg.dp}")
print()

# Run comparison
rows = compare_precision(model, hw, parallel_cfg, rl_cfg, recipes)

# Print table
col_w = 12
header = (
    f"{'Recipe':<8} {'Step(s)':>{col_w}} {'Speedup':>{col_w}} "
    f"{'CommRed%':>{col_w}} {'PeakMem(GB)':>{col_w}} {'Feasible':>{col_w}}"
)
sep = "-" * len(header)
print(header)
print(sep)
for r in rows:
    print(
        f"{r['name']:<8} "
        f"{r['step_seconds']:>{col_w}.4f} "
        f"{r['speedup_vs_bf16']:>{col_w}.3f} "
        f"{r['comm_reduction_pct']:>{col_w}.1f} "
        f"{r['peak_memory_gb']:>{col_w}.1f} "
        f"{'Yes' if r['feasible'] else 'No':>{col_w}}"
    )
print()

# Show fabric-level exposed comm for fp8
fp8_row = next((r for r in rows if r["name"] == "fp8"), None)
if fp8_row:
    print("fp8 exposed comm by fabric:")
    for fabric, secs in sorted(fp8_row["exposed_comm_by_fabric"].items()):
        print(f"  {fabric}: {secs * 1000:.2f} ms")
