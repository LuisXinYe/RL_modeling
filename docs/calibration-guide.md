# Calibration Guide

llm-perf uses a roofline-based cost model. Calibration coefficients bridge the gap between theoretical peak performance and what the hardware actually achieves. This guide explains each coefficient, how to measure it, and when to update.

---

## What Calibration Does

The roofline model computes operator time as:

```
time = max(compute_time, memory_time)
```

where:
- `compute_time = FLOPs / (peak_tflops * compute_efficiency * 1e12)`
- `memory_time = bytes / (hbm_bandwidth * memory_efficiency * 1e12)`

Without calibration (all efficiencies = 1.0), the model assumes the hardware runs at theoretical peak -- which never happens in practice. Calibration coefficients scale these peaks down to realistic achievable throughput.

Communication time is similarly scaled:

```
comm_time = message_bytes / (link_bandwidth * comm_efficiency)
```

---

## Coefficient Reference

All coefficients live in the `calibration` block of your hardware YAML file.

### `compute_eff_large_gemm`

**Default:** 0.50 | **Range:** 0.3 -- 0.8

What it represents: The fraction of peak BF16 TFLOPS that large matrix multiplications (GEMMs) actually achieve. Large GEMMs dominate the runtime of attention projections (QKV, output) and FFN layers.

**Why it matters most:** Large GEMMs account for 70--90% of total FLOPs in transformer training and inference. A 10% change in this coefficient shifts the total predicted time by 7--9%.

**How to measure:**
1. Run a GEMM benchmark with representative matrix sizes from your model. For a model with hidden_size=8192 and intermediate_size=29568:
   - Attention QKV projection: `[batch*seq, 8192] x [8192, 8192]`
   - FFN gate/up: `[batch*seq, 8192] x [8192, 29568]`
2. Measure achieved TFLOPS: `2 * M * N * K / time_seconds / 1e12`
3. Divide by `peak_tflops_bf16` to get the efficiency ratio.

**Typical values:**
- Ascend 910C: 0.45 -- 0.55
- A100/H100: 0.55 -- 0.70
- Poorly tuned kernels: 0.30 -- 0.40

### `compute_eff_small_op`

**Default:** 0.20 | **Range:** 0.05 -- 0.40

What it represents: Efficiency for small or non-GEMM operators (LayerNorm, RMSNorm, activation functions, softmax, element-wise ops). These are typically memory-bandwidth-bound but some compute still applies.

**Why it matters:** Small ops contribute relatively little to total time in large models but can become significant in decode (small batch) or with many auxiliary ops.

**How to measure:**
1. Profile a single transformer layer and identify non-GEMM operators.
2. Measure their combined runtime and FLOPs.
3. Compute efficiency as above.

**Typical values:**
- Most hardware: 0.10 -- 0.25
- These ops are almost always bandwidth-bound, so compute efficiency is low by nature.

### `memory_efficiency`

**Default:** 0.70 | **Range:** 0.50 -- 0.90

What it represents: The fraction of peak HBM bandwidth actually achieved. This affects all memory-bound operations: decode attention, KV cache reads, weight loading for small-batch inference.

**Why it matters:** Autoregressive decode is almost entirely memory-bandwidth-bound. This coefficient directly controls predicted decode throughput.

**How to measure:**
1. Run a memory bandwidth benchmark (e.g., a large vector copy or stream benchmark).
2. Compare achieved bandwidth against the spec sheet value (`hbm_bandwidth_tb_s`).
3. Alternatively, profile decode-phase attention and compare actual bandwidth utilization.

**Typical values:**
- Ascend 910C: 0.65 -- 0.75
- A100 (80GB): 0.70 -- 0.85
- H100: 0.75 -- 0.90

### `comm_efficiency`

**Default:** 0.70 | **Range:** 0.50 -- 0.90

What it represents: The fraction of peak interconnect bandwidth achieved for collective communication (AllReduce, AllGather, ReduceScatter).

**Why it matters:** With TP >= 4, communication overhead becomes significant. This coefficient determines how much time the model spends waiting on collectives between layers.

**How to measure:**
1. Run an AllReduce benchmark (e.g., NCCL/HCCL tests) with message sizes typical of your model:
   - TP AllReduce: `2 * batch * seq * hidden_size * dtype_bytes` per layer
   - DP AllReduce: total gradient size / DP degree
2. Compute effective bandwidth and divide by the link bandwidth (`intra_node_bw_gb_s` for TP within a node, `inter_node_bw_gb_s` for cross-node DP).

**Typical values:**
- Intra-node (NVLink/HCCS): 0.70 -- 0.85
- Inter-node (RDMA): 0.50 -- 0.70
- CloudMatrix 384 (high-bandwidth fabric): 0.75 -- 0.85

---

## Which Coefficients Matter Most

In rough order of impact on total predicted time:

1. **`compute_eff_large_gemm`** -- Dominates training time and prefill time. This is the single most important coefficient.
2. **`memory_efficiency`** -- Dominates decode time. Critical for generation-heavy workloads.
3. **`comm_efficiency`** -- Important when TP is high (>= 4) or DP crosses node boundaries.
4. **`compute_eff_small_op`** -- Least impactful for large models. Worth tuning for small models or decode-heavy configs.

**Recommendation:** Start by calibrating `compute_eff_large_gemm` and `memory_efficiency`. These two alone will get you within 15--20% of reality. Add `comm_efficiency` calibration for multi-node setups.

---

## When to Recalibrate

Recalibrate when:

1. **New hardware** -- Each accelerator has different efficiency characteristics. Always benchmark new hardware.
2. **Major framework/driver update** -- Kernel implementations change between framework versions (e.g., PyTorch 2.x fused attention, new CANN versions).
3. **New operator kernels** -- If you switch from standard attention to FlashAttention, or enable custom GEMM kernels, efficiencies will change.
4. **Predictions are consistently off by > 20%** -- This indicates the calibration no longer matches reality. Re-benchmark and update.
5. **Switching from single-node to multi-node** -- Communication efficiency often drops significantly when crossing node boundaries.

**What does NOT require recalibration:**
- Changing model size (same hardware, same kernels)
- Changing batch size (efficiencies are roughly stable across batch sizes for large GEMMs)
- Changing parallelism strategy (the model accounts for parallelism directly)

---

## Quick Calibration Workflow

```bash
# 1. Run GEMM benchmark (example: measure 8192x8192 matmul)
python benchmarks/gemm_bench.py --m 4096 --n 8192 --k 8192 --dtype bf16

# 2. Run bandwidth benchmark
python benchmarks/bandwidth_bench.py --size 1GB

# 3. Run AllReduce benchmark (8 devices)
python benchmarks/allreduce_bench.py --devices 8 --size 64MB

# 4. Update your hardware YAML
# configs/hardware/my_hardware.yaml
calibration:
  compute_eff_large_gemm: <measured>
  compute_eff_small_op: <measured or estimated>
  memory_efficiency: <measured>
  comm_efficiency: <measured>
```

If you do not have benchmark infrastructure, the defaults (50% compute, 70% memory/comm) are deliberately conservative. Your actual hardware will likely perform better, meaning llm-perf predictions will be pessimistic (overestimate time, underestimate throughput). This is generally safer than optimistic predictions.
