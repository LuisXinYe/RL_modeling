# Result Interpretation Guide

This document explains how to read and act on the output of `llm-perf`.

---

## TargetReport Fields

The main output of `LLMPerformanceModel.derive_targets()` is a `TargetReport` (defined in `src/llm_perf/report.py`).

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `epoch_time_hours` | `float` | hours | Predicted wall-clock time for one RL epoch |
| `within_budget` | `bool` | -- | `True` if `epoch_time_hours <= time_budget_hours` (always `True` if no budget set) |
| `bottleneck` | `str` | -- | Which phase dominates: `"GENERATION"`, `"TRAINING"`, or `"BALANCED"` |
| `bottleneck_slack` | `float` | ratio | How much slower the bottleneck phase is relative to the other (0.0 = balanced) |
| `gen_tps_target` | `float` | tokens/s | Required generation throughput (tokens per second, across all devices) |
| `train_tps_target` | `float` | tokens/s | Required training throughput (tokens per second, across all devices) |
| `gen_samples_per_sec` | `float` | samples/s | Generation throughput in samples per second |
| `train_samples_per_sec` | `float` | samples/s | Training throughput in samples per second |
| `gen_time_hours` | `float` | hours | Total generation phase time |
| `train_time_hours` | `float` | hours | Total training phase time |
| `memory` | `MemoryProfile` | -- | Per-device memory breakdown (see below) |
| `gen_parallel` | `ParallelismConfig` | -- | Parallelism config used for generation |
| `train_parallel` | `ParallelismConfig` | -- | Parallelism config used for training |
| `feasible` | `bool` | -- | `True` only if `within_budget` AND memory is feasible for both phases |

---

## MemoryProfile Fields

Per-device memory breakdown (all values in GB).

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `weight_gb` | `float` | GB | Model weight memory (after TP/PP sharding) |
| `optimizer_gb` | `float` | GB | Optimizer state memory (Adam: 12 bytes/param; reduced by ZeRO stage >= 1) |
| `activation_peak_gb` | `float` | GB | Peak activation memory during training forward pass |
| `kv_cache_gb` | `float` | GB | KV cache memory during generation (per device) |
| `ref_model_gb` | `float` | GB | Reference model memory (0 if `reference_model=False` or `ref_offload_cpu=True`) |
| `total_train_gb` | `float` | GB | Total training memory = weight + optimizer + activation + ref_model |
| `total_gen_gb` | `float` | GB | Total generation memory = weight + kv_cache |
| `usable_hbm_gb` | `float` | GB | Usable HBM per device (capacity * usable_ratio) |
| `train_feasible` | `bool` | -- | `True` if `total_train_gb < usable_hbm_gb` |
| `gen_feasible` | `bool` | -- | `True` if `total_gen_gb < usable_hbm_gb` |

---

## Decision Tree

Use this flowchart to decide your next step based on the report:

```
Is feasible = True?
  |
  +-- YES --> Proceed. Use TPS targets as benchmarks for your team.
  |
  +-- NO --> Check why:
        |
        +-- OOM (train_feasible=False)?
        |     |
        |     +-- Reduce train_micro_batch_size
        |     +-- Increase TP (spreads weights + activations across more devices)
        |     +-- Enable recompute_attention or full_recomputation (trades compute for memory)
        |     +-- Enable optimizer_offload (moves optimizer states to CPU)
        |     +-- Enable ZeRO stage 1+ (shards optimizer states across DP ranks)
        |     +-- If reference_model is on: set ref_offload_cpu=True
        |
        +-- OOM (gen_feasible=False)?
        |     |
        |     +-- Reduce gen_batch_size (reduces KV cache)
        |     +-- Increase TP (shards KV heads across more devices)
        |     +-- For MLA models: KV cache is already compressed, focus on batch size
        |     +-- For SWA layers: KV cache is bounded by window_size, check other layers
        |
        +-- OVER TIME (within_budget=False)?
              |
              +-- Add more devices (increases DP, reduces total batches)
              +-- Increase gen_batch_size (fewer generation batches)
              +-- If bottleneck is GENERATION: focus on generation throughput
              +-- If bottleneck is TRAINING: increase gradient_accumulation_steps
              +-- Consider colocated=False if currently True (enables overlap)
```

---

## Bottleneck Interpretation

### What the bottleneck field means

The RL training pipeline has two main phases:

1. **GENERATION** -- The model generates responses for all prompts (prefill + autoregressive decode).
2. **TRAINING** -- The model trains on the generated (prompt, response) pairs.

When `colocated=False` (separate device pools), the two phases can overlap. The epoch time is approximately `max(gen_time, train_time) + startup_overhead`. The slower phase is the **bottleneck**.

When `colocated=True` (same devices), the phases run serially and epoch time is `gen_time + train_time`.

### Bottleneck values

| Value | Meaning |
|-------|---------|
| `GENERATION` | Generation takes longer than training. The training cluster is idle waiting for data. |
| `TRAINING` | Training takes longer than generation. The generation cluster finishes early and waits. |
| `BALANCED` | Both phases take approximately the same time (ideal). |

### Reading the slack ratio

The `bottleneck_slack` is defined as:

```
slack = (slower_phase_time / faster_phase_time) - 1
```

Examples:
- `slack = 0.0` -- Perfectly balanced.
- `slack = 0.5` -- The bottleneck phase takes 50% longer than the other. The faster cluster is idle for 33% of the epoch.
- `slack = 2.0` -- The bottleneck phase takes 3x longer. The faster cluster is idle for 67% of the epoch.

**Rule of thumb:** A slack ratio above 0.5 indicates significant resource waste. Consider rebalancing device allocation between generation and training pools.

### What to do about each bottleneck

**GENERATION bottleneck:**
- Increase `gen_batch_size` (more tokens decoded per step, better GPU utilization)
- Allocate more devices to generation (increase gen DP)
- Enable speculative decoding (`use_speculative_decoding=True`) if the model has MTP heads
- Reduce `max_response_len` if possible

**TRAINING bottleneck:**
- Increase `gradient_accumulation_steps` (larger effective batch, fewer steps)
- Allocate more devices to training (increase train DP)
- Reduce PP degree if bubble overhead is significant
- Check if recomputation/offload penalties are stacking up

---

## Example Output

```
============================================================
          LLM Performance Report
============================================================
 Epoch time:        2.45 hours  [FEASIBLE]
 Bottleneck:        GENERATION (slack: 23.5%)
------------------------------------------------------------
 Generation:
   TPS target:      185,000 tokens/s
   Samples/s:       12.50
   Time:            1.78 hours
------------------------------------------------------------
 Training:
   TPS target:      320,000 tokens/s
   Samples/s:       8.30
   Time:            1.44 hours
------------------------------------------------------------
 Memory:
   Train: 89.2/108.8 GB  [OK]
   Gen:   72.5/108.8 GB  [OK]
============================================================
```

Reading this report:
- The epoch completes in 2.45 hours and is feasible.
- Generation is the bottleneck with 23.5% slack -- generation takes ~24% longer than training.
- The generation cluster must sustain 185K tokens/s; the training cluster must sustain 320K tokens/s.
- Memory fits on both clusters with headroom.
