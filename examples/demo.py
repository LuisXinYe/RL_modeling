#!/usr/bin/env python3
"""llm-perf Python API demo: Llama-3.1-8B RL training performance modeling.

This demo uses the new runtime config system. For other scenarios, see:
  - examples/demo_rl.py         (RL post-training)
  - examples/demo_inference.py  (standalone inference)
  - examples/demo_pretraining.py (standalone pretraining)
"""

from pathlib import Path

from llm_perf.config import load_model_config, load_hardware_config, load_runtime_config
from llm_perf.model import LLMPerformanceModel
from llm_perf.report import format_table

ROOT = Path(__file__).resolve().parent.parent

# Load from runtime config (combines model, hardware, parallelism, RL params)
rt = load_runtime_config(ROOT / "configs/runtime/llama3_1_8b_8x_910c.yaml")

# Resolve model and hardware from runtime config references
_MODEL_MAP = {
    "Llama-3.1-8B": "llama3_1_8b",
    "Qwen2.5-72B": "qwen2_5_72b",
    "Mistral-7B": "mistral_7b",
    "Qwen3-235B-A22B": "qwen3_235b_moe",
    "DeepSeek-V3-671B": "deepseekv3_671b",
}
_HW_MAP = {
    "Ascend 910B": "ascend_910b",
    "Ascend 910C": "ascend_910c",
    "CloudMatrix 384": "cloudmatrix_384",
}
model_stem = _MODEL_MAP.get(rt.model, rt.model)
hw_stem = _HW_MAP.get(rt.hardware, rt.hardware)
model = load_model_config(ROOT / "configs" / "models" / f"{model_stem}.yaml")
hw = load_hardware_config(ROOT / "configs" / "hardware" / f"{hw_stem}.yaml")
perf = LLMPerformanceModel(model, hw)

print(f"Model: {model.name}")
print(f"  hidden_size: {model.hidden_size}")
print(f"  num_layers: {model.num_layers}")
print(f"  actual layers: {len(model.get_layers())}")
dense = sum(1 for l in model.get_layers() if l.ffn == "SwiGLU")
moe = sum(1 for l in model.get_layers() if l.ffn == "MoE")
print(f"  Dense layers: {dense}, MoE layers: {moe}")
print(f"Hardware: {hw.name}")
print(f"  peak_tflops_bf16: {hw.peak_tflops_bf16}")
print(f"  hbm_capacity_gb: {hw.hbm_capacity_gb}")
print()

# Parallelism from runtime config
train_p = rt.parallelism
gen_p = rt.gen_parallelism or rt.parallelism
ref_p = rt.ref_parallelism or rt.parallelism

# RL config from runtime config
rl_cfg = rt.rl

print(f"Generation: TP={gen_p.tp} PP={gen_p.pp} EP={gen_p.ep} DP={gen_p.dp}")
print(f"Training:   TP={train_p.tp} PP={train_p.pp} EP={train_p.ep} DP={train_p.dp} CP={train_p.cp}")
print(f"Reference:  TP={ref_p.tp} PP={ref_p.pp} EP={ref_p.ep} DP={ref_p.dp}")
print()

# --- Run prediction ---
print("=" * 60)
print("Running performance prediction...")
print("=" * 60)
report = perf.derive_targets(
    total_devices=rt.total_devices,
    rl_cfg=rl_cfg,
    gen_parallel=gen_p,
    train_parallel=train_p,
    ref_parallel=ref_p,
)
print(format_table(report))