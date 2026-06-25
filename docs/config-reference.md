# Configuration Reference

This document describes every configuration class used by llm-perf.
All configs are defined in `src/llm_perf/config.py` and loaded from YAML files.

---

## ModelConfig

Top-level model architecture definition.

| Field | Type | Default | Unit | Description | Constraints |
|-------|------|---------|------|-------------|-------------|
| `name` | `str` | *required* | -- | Display name for the model | -- |
| `hidden_size` | `int` | *required* | elements | Model hidden dimension (d_model) | -- |
| `vocab_size` | `int` | *required* | tokens | Vocabulary size | -- |
| `num_layers` | `int` | *required* | -- | Total transformer layers | -- |
| `dtype` | `str` | `"bf16"` | -- | Weight data type | One of: `bf16`, `fp16`, `fp32`, `fp8` |
| `default_layer` | `LayerConfig` | `None` | -- | Layer config applied to all layers | Provide this OR `layers` |
| `layers` | `List[LayerConfig]` | `None` | -- | Per-layer config list (for hybrid architectures) | Length must equal `num_layers` |
| `auxiliary` | `dict` | `None` | -- | Extra modules, e.g. `{"mtp_depth": 1}` | -- |

**Derived properties:**
- `dtype_bytes` -- bytes per element: bf16/fp16 = 2, fp32 = 4, fp8 = 1
- `get_layers()` -- returns the list of `LayerConfig` for all layers (from `layers` if set, otherwise replicates `default_layer`)

**Validation:** You must provide either `default_layer` or `layers`. If neither is set, `get_layers()` raises `ValueError`.

### Example YAML (ModelConfig)

```yaml
name: "Qwen2.5-72B"
hidden_size: 8192
vocab_size: 152064
num_layers: 80
dtype: bf16

default_layer:
  attention: GQA
  num_heads: 64
  num_kv_heads: 8
  head_dim: 128
  ffn: SwiGLU
  intermediate_size: 29568
```

See `configs/models/_template.yaml` for a fully commented template.

---

## LayerConfig

Per-layer architecture parameters. Supports multiple attention and FFN variants.

### Attention Fields

| Field | Type | Default | Unit | Description | Constraints |
|-------|------|---------|------|-------------|-------------|
| `attention` | `str` | `"GQA"` | -- | Attention variant | One of: `MHA`, `GQA`, `MLA`, `SWA`, `Mamba` |
| `num_heads` | `int` | `32` | -- | Number of query attention heads | Must be divisible by TP degree |
| `num_kv_heads` | `int` | `8` | -- | Number of key/value heads | GQA: < `num_heads`; MHA: = `num_heads` |
| `head_dim` | `int` | `128` | elements | Dimension per attention head | -- |

### MLA-specific Fields (only when `attention: MLA`)

| Field | Type | Default | Unit | Description | Constraints |
|-------|------|---------|------|-------------|-------------|
| `kv_compression_dim` | `int` | `0` | elements | KV latent compression dimension (d_c) | Set > 0 for MLA |
| `query_compression_dim` | `int` | `0` | elements | Query latent compression dimension (d'_c) | Set > 0 for MLA |
| `rope_dim` | `int` | `0` | elements | Decoupled RoPE dimension (d_R) | Set > 0 for MLA |

### SWA-specific Fields (only when `attention: SWA`)

| Field | Type | Default | Unit | Description | Constraints |
|-------|------|---------|------|-------------|-------------|
| `window_size` | `int` | `0` | tokens | Sliding window size | 0 = full attention |

### FFN Fields

| Field | Type | Default | Unit | Description | Constraints |
|-------|------|---------|------|-------------|-------------|
| `ffn` | `str` | `"SwiGLU"` | -- | FFN variant | One of: `SwiGLU`, `MoE` |
| `intermediate_size` | `int` | `11008` | elements | FFN hidden dimension (for SwiGLU) | -- |

### MoE-specific Fields (only when `ffn: MoE`)

| Field | Type | Default | Unit | Description | Constraints |
|-------|------|---------|------|-------------|-------------|
| `num_experts` | `int` | `1` | -- | Total number of MoE experts | Must be divisible by EP degree |
| `num_shared_experts` | `int` | `0` | -- | Number of always-active shared experts | -- |
| `top_k` | `int` | `1` | -- | Experts activated per token | -- |
| `expert_intermediate_size` | `int` | `0` | elements | Per-expert FFN hidden dimension | Required when `ffn: MoE` |
| `shared_intermediate_size` | `int` | `0` | elements | Shared expert FFN hidden dimension | Required when `num_shared_experts > 0` |

### Residual Fields

| Field | Type | Default | Unit | Description | Constraints |
|-------|------|---------|------|-------------|-------------|
| `residual` | `str` | `"standard"` | -- | Residual connection type | One of: `standard`, `mHC` |
| `mhc_expansion` | `int` | `4` | -- | Expansion factor for mHC residual | Only used when `residual: mHC` |

---

## HardwareConfig

Hardware capability description for a single device.

| Field | Type | Default | Unit | Description | Constraints |
|-------|------|---------|------|-------------|-------------|
| `name` | `str` | *required* | -- | Hardware display name | -- |
| `peak_tflops_bf16` | `float` | *required* | TFLOPS | Peak BF16 compute throughput | -- |
| `hbm_capacity_gb` | `float` | *required* | GB | Total HBM capacity per device | -- |
| `hbm_bandwidth_tb_s` | `float` | *required* | TB/s | HBM bandwidth | -- |
| `hbm_usable_ratio` | `float` | `0.85` | ratio | Fraction of HBM available after framework overhead | 0.0 -- 1.0 |
| `intra_node_bw_gb_s` | `float` | `400` | GB/s | Intra-node interconnect bandwidth | -- |
| `inter_node_bw_gb_s` | `float` | `100` | GB/s | Inter-node interconnect bandwidth | -- |
| `inter_node_latency_us` | `float` | `5` | microseconds | Inter-node communication latency | -- |
| `devices_per_node` | `int` | `8` | -- | Number of devices per node | -- |
| `calibration` | `CalibrationConfig` | *(see below)* | -- | Calibration coefficients | -- |

**Derived properties:**
- `usable_hbm_gb` = `hbm_capacity_gb * hbm_usable_ratio`

### Example YAML (HardwareConfig)

```yaml
# configs/hardware/ascend_910c.yaml
name: "Ascend 910C"
peak_tflops_bf16: 800
hbm_capacity_gb: 128
hbm_bandwidth_tb_s: 3.2
hbm_usable_ratio: 0.85
intra_node_bw_gb_s: 400
inter_node_bw_gb_s: 100
inter_node_latency_us: 5
devices_per_node: 8
calibration:
  compute_eff_large_gemm: 0.50
  compute_eff_small_op: 0.20
  memory_efficiency: 0.70
  comm_efficiency: 0.70
```

---

## CalibrationConfig

Efficiency coefficients that scale theoretical peak performance down to achievable performance.
These are embedded inside `HardwareConfig` under the `calibration` key.

| Field | Type | Default | Unit | Description | Constraints |
|-------|------|---------|------|-------------|-------------|
| `compute_eff_large_gemm` | `float` | `0.50` | ratio | Achieved fraction of peak TFLOPS for large GEMMs | 0.0 -- 1.0 |
| `compute_eff_small_op` | `float` | `0.20` | ratio | Achieved fraction of peak TFLOPS for small/non-GEMM ops | 0.0 -- 1.0 |
| `memory_efficiency` | `float` | `0.70` | ratio | Achieved fraction of peak HBM bandwidth | 0.0 -- 1.0 |
| `comm_efficiency` | `float` | `0.70` | ratio | Achieved fraction of peak interconnect bandwidth | 0.0 -- 1.0 |

See [calibration-guide.md](calibration-guide.md) for how to measure and tune these values.

---

## WorkloadConfig

RL training workload parameters.

| Field | Type | Default | Unit | Description | Constraints |
|-------|------|---------|------|-------------|-------------|
| `total_prompts` | `int` | *required* | prompts | Number of prompts per epoch | -- |
| `group_size` | `int` | `8` | responses/prompt | Number of responses generated per prompt (GRPO) | -- |
| `avg_prompt_len` | `int` | `512` | tokens | Average prompt length | -- |
| `avg_response_len` | `int` | `2048` | tokens | Average response length | -- |
| `max_response_len` | `int` | `4096` | tokens | Maximum response length (generation cutoff) | -- |
| `std_response_len` | `int` | `None` | tokens | Std dev of response length (enables Gumbel batch-max estimation) | -- |
| `train_micro_batch_size` | `int` | `4` | samples | Micro-batch size for training | -- |
| `gradient_accumulation_steps` | `int` | `1` | steps | Gradient accumulation steps | -- |
| `gen_batch_size` | `int` | `64` | samples | Batch size for generation (per DP rank) | -- |
| `reference_model` | `bool` | `True` | -- | Whether a reference model is kept in memory | -- |
| `ref_offload_cpu` | `bool` | `False` | -- | Offload reference model to CPU memory | -- |
| `colocated` | `bool` | `False` | -- | Generation and training on the same GPUs (serial) | -- |
| `use_speculative_decoding` | `bool` | `False` | -- | Enable speculative decoding via MTP heads | -- |
| `mtp_acceptance_len` | `int` | `None` | tokens | Expected acceptance length for speculative decoding | Defaults to `mtp_depth` if not set |

**Derived properties:**
- `total_responses` = `total_prompts * group_size`

---

## ParallelismConfig

Parallelism and memory optimization settings. Applied separately for generation and training.

| Field | Type | Default | Unit | Description | Constraints |
|-------|------|---------|------|-------------|-------------|
| `tp` | `int` | `1` | -- | Tensor Parallelism degree | >= 1; `num_heads` must be divisible by `tp` |
| `pp` | `int` | `1` | -- | Pipeline Parallelism degree | >= 1 |
| `dp` | `int` | `1` | -- | Data Parallelism degree | >= 1 |
| `ep` | `int` | `1` | -- | Expert Parallelism degree (MoE only) | >= 1; `num_experts` must be divisible by `ep` |
| `cp` | `int` | `1` | -- | Context Parallelism degree | >= 1 |
| `cp_type` | `str` | `"ring"` | -- | Context parallelism communication pattern | -- |
| `sp` | `bool` | `False` | -- | Sequence Parallelism (replaces AllReduce with AllGather + ReduceScatter) | -- |
| `zero_stage` | `int` | `0` | -- | ZeRO optimization stage | 0, 1, 2, or 3 |
| `pp_schedule` | `str` | `"1f1b"` | -- | Pipeline schedule | -- |
| `pp_virtual_stages` | `int` | `1` | -- | Virtual pipeline stages (interleaved schedule) | -- |
| `recompute_attention` | `bool` | `False` | -- | Recompute attention in backward pass (saves activation memory) | +5% time penalty |
| `full_recomputation` | `bool` | `False` | -- | Full activation recomputation | +30% time penalty |
| `optimizer_offload` | `bool` | `False` | -- | Offload optimizer states to CPU | +15% time penalty |
| `activation_offload` | `bool` | `False` | -- | Offload activations to CPU | +10% time penalty |

**Derived properties:**
- `total_devices` = `tp * pp * dp * ep`

**Validation:** All parallelism degrees (`tp`, `pp`, `dp`, `ep`, `cp`) must be >= 1.

---

## Phase (Enum)

Execution phase selector for the cost model.

| Value | Description |
|-------|-------------|
| `PREFILL` | Generation prefill (prompt processing, compute-bound) |
| `DECODE` | Generation autoregressive decode (memory-bandwidth-bound) |
| `TRAIN_FWD` | Training forward pass |
| `TRAIN_BWD` | Training backward pass |

---

## Common Pitfalls

1. **Missing `default_layer` and `layers`** -- `ModelConfig.get_layers()` will raise `ValueError`. You must set one of them.

2. **`num_heads` not divisible by `tp`** -- This causes a `ValueError` at simulation time. Choose TP degree that evenly divides the number of attention heads.

3. **`num_experts` not divisible by `ep`** -- For MoE models, expert parallelism degree must evenly divide the total expert count.

4. **Forgetting `calibration` block** -- Defaults are conservative (50% compute efficiency). If your hardware is well-tuned, you will underestimate performance. Run benchmarks and update calibration values.

5. **`hbm_usable_ratio` too high** -- Framework overhead (CUDA/CANN context, memory allocator fragmentation) typically consumes 10--20% of HBM. The default of 0.85 is reasonable for most setups.

6. **`colocated: true` with separate device pools** -- When `colocated` is true, epoch time is `gen_time + train_time` (serial). Only set this if generation and training share the same GPUs.

7. **Offload penalties stack multiplicatively** -- Enabling both `optimizer_offload` and `activation_offload` together with `full_recomputation` applies 1.30 * 1.15 * 1.10 = 1.65x penalty to each training step.
