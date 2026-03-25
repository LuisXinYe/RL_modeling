# rl-perf: RL Training Performance Modeling

Given model + hardware + RL config, predict epoch time and derive TPS targets for inference and training teams.

## Features

- **Targets derivation**: model + devices + data + time budget → gen/train TPS targets
- **Feasibility check**: will it fit in memory? how long will it take?
- **What-if analysis**: change group_size, parallelism, hardware → compare results
- **Sensitivity sweep**: scan parameter ranges to find optimal configs

## Supported Architectures

| Component | Variants |
|-----------|----------|
| Attention | MHA, GQA, MLA (DeepSeek), SWA (Mistral) |
| FFN | SwiGLU, MoE (routed + shared experts) |
| Residual | Standard, mHC (manifold-constrained HyperConnections) |
| Parallelism | TP, PP (1F1B), DP/ZeRO, EP, CP, SP |

## Demo Models

| Model | Type | Attention | FFN |
|-------|------|-----------|-----|
| Llama 3.1 8B | Dense | GQA | SwiGLU |
| Qwen2.5 72B | Dense | GQA | SwiGLU |
| Mistral 7B | Dense | GQA+SWA | SwiGLU |
| Qwen3 235B-A22B | MoE | GQA | MoE (128E, top-8) |
| DeepSeek V3 671B | MoE | MLA | MoE (256E, top-8+1shared) |

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick Start

```bash
# Derive TPS targets: Llama 8B on 64x Ascend 910C, 10K prompts, 24h budget
rl-perf targets \
  --model configs/models/llama3_1_8b.yaml \
  --hardware 910C \
  --devices 64 \
  --prompts 10000 \
  --group-size 8 \
  --time-budget 24

# Feasibility check for a larger model
rl-perf check \
  --model configs/models/qwen2_5_72b.yaml \
  --hardware 910C \
  --devices 128 \
  --prompts 10000 \
  --tp 8 --pp 4
```

## Python API

```python
from rl_perf.config import load_model_config, load_hardware_config, RLConfig, ParallelismConfig
from rl_perf.model import RLPerformanceModel

model = load_model_config("configs/models/llama3_1_8b.yaml")
hw = load_hardware_config("configs/hardware/ascend_910c.yaml")
perf = RLPerformanceModel(model, hw)

rl_cfg = RLConfig(total_prompts=10000, group_size=8)
gen_p = ParallelismConfig(tp=8, dp=8)
train_p = ParallelismConfig(tp=8, dp=8)

report = perf.derive_targets(64, rl_cfg, gen_p, train_p, time_budget_hours=24)
print(f"Epoch: {report.epoch_time_hours:.2f}h | Gen TPS: {report.gen_tps_target:,.0f} | Train TPS: {report.train_tps_target:,.0f}")
```

## Custom Model

Copy `configs/models/_template.yaml` and fill in your model's parameters. See the template for detailed field descriptions.

## Tutorials

| Resource | Description |
|----------|-------------|
| [`examples/demo.py`](examples/demo.py) | Quick demo: prediction, what-if comparison, sensitivity sweep |
| [`notebooks/01_quick_start.ipynb`](notebooks/01_quick_start.ipynb) | First prediction in 5 minutes |
| [`notebooks/02_what_if_analysis.ipynb`](notebooks/02_what_if_analysis.ipynb) | Compare parallelism strategies and hardware |
| [`notebooks/03_moe_scaling.ipynb`](notebooks/03_moe_scaling.ipynb) | MoE vs Dense scaling with EP trade-offs |

## Architecture

```
YAML Config → ops.py (roofline per op) → builder.py (parallelism + DAG)
           → simulator.py (multi-stream sim) → pipeline.py (gen/train pipeline)
           → TargetReport (TPS targets + memory profile)
```

See [docs/architecture.md](docs/architecture.md) for a detailed deep-dive.

## Documentation

- [Architecture deep-dive](docs/architecture.md) — Three-layer design, roofline model, pipeline simulation
- [Config reference](docs/config-reference.md) — All configuration fields, constraints, and defaults
- [Result interpretation](docs/result-interpretation.md) — How to read reports and make decisions
- [Calibration guide](docs/calibration-guide.md) — Measuring and tuning calibration coefficients
- [Troubleshooting](docs/troubleshooting.md) — Common errors and fixes

## Tests

```bash
pytest tests/ -v          # 139 tests
```

Requires Python >= 3.10.

## Accuracy

- Without benchmark calibration: ~50% (theoretical SOL mode)
- With benchmark calibration: <30% target (calibrate via configs/hardware/*.yaml)
