#!/usr/bin/env python3
"""rl-perf Python API demo: single prediction, what-if, sensitivity sweep."""

from pathlib import Path
from rl_perf.config import load_model_config, load_hardware_config, RLConfig, ParallelismConfig
from rl_perf.model import RLPerformanceModel
from rl_perf.report import format_table

ROOT = Path(__file__).resolve().parent.parent
model = load_model_config(ROOT / "configs/models/llama3_1_8b.yaml")
hw = load_hardware_config(ROOT / "configs/hardware/ascend_910c.yaml")
perf = RLPerformanceModel(model, hw)

rl_cfg = RLConfig(total_prompts=10_000, group_size=8, avg_prompt_len=512, avg_response_len=2048)
gen_p = ParallelismConfig(tp=2, dp=4)
train_p = ParallelismConfig(tp=2, dp=4)

# --- 1. Single prediction ---
report = perf.derive_targets(total_devices=8, rl_cfg=rl_cfg,
                             gen_parallel=gen_p, train_parallel=train_p,
                             time_budget_hours=24)
print(format_table(report))

# --- 2. What-if: vary TP degree ---
print("\n  TP | epoch_time(h) | gen_tps | train_tps | feasible")
print("  ---+---------------+---------+-----------+---------")
for tp in [1, 2, 4, 8]:
    gp = ParallelismConfig(tp=tp, dp=max(1, 8 // tp))
    tp_train = ParallelismConfig(tp=tp, dp=max(1, 8 // tp))
    r = perf.derive_targets(8, rl_cfg, gp, tp_train)
    print(f"  {tp:2d} | {r.epoch_time_hours:13.2f} | {r.gen_tps_target:7.0f} | {r.train_tps_target:9.0f} | {r.feasible}")

# --- 3. Sensitivity sweep: group_size → epoch_time bar chart ---
group_sizes = [2, 4, 8, 16, 32]
reports = perf.sensitivity(rl_cfg, "group_size", group_sizes, 8, gen_p, train_p)
max_hours = max(r.epoch_time_hours for r in reports)
bar_width = 40

print("\ngroup_size -> epoch_time (hours)")
for gs, r in zip(group_sizes, reports):
    filled = int(bar_width * r.epoch_time_hours / max_hours) if max_hours > 0 else 0
    bar = "\u2588" * filled
    print(f"  {gs:2d}  {bar:<{bar_width}}  {r.epoch_time_hours:.1f}h")
