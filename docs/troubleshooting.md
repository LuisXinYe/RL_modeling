# Troubleshooting

Common issues, symptoms, and fixes for rl-perf.

---

## `command not found: rl-perf`

**Symptom:**
```
zsh: command not found: rl-perf
```

**Cause:** The package is not installed in your active Python environment.

**Fix:**
```bash
source .venv/bin/activate
pip install -e ".[dev]"
```

Make sure you are in the project root directory and using the correct virtual environment.

---

## `FileNotFoundError`

**Symptom:**
```
Error: file not found: [Errno 2] No such file or directory: 'configs/models/my_model.yaml'
```

**Cause:** The config file path is wrong, or you are running the command from the wrong directory.

**Fixes:**
1. Check that the file exists at the specified path.
2. Use absolute paths, or run from the project root.
3. For hardware configs, use the built-in shortnames instead of full paths:
   - `910C` resolves to `configs/hardware/ascend_910c.yaml`
   - `CM384` resolves to `configs/hardware/cloudmatrix_384.yaml`

```bash
# Using shortname
rl-perf targets -m configs/models/qwen2_5_72b.yaml -hw 910C -d 64 -p 10000

# Using full path
rl-perf targets -m configs/models/qwen2_5_72b.yaml -hw configs/hardware/ascend_910c.yaml -d 64 -p 10000
```

---

## `ValidationError`

**Symptom:**
```
Error: invalid config: 1 validation error for HardwareConfig
peak_tflops_bf16
  field required (type=value_error.missing)
```

**Cause:** A required field is missing or has the wrong type in your YAML file.

**Fixes:**
1. Compare your YAML against the templates:
   - Model: `configs/models/_template.yaml`
   - Hardware: `configs/hardware/ascend_910c.yaml`
2. Check for typos in field names (YAML is case-sensitive).
3. Make sure numeric fields are not quoted as strings.
4. See [config-reference.md](config-reference.md) for the full list of required and optional fields.

Common mistakes:
```yaml
# WRONG: quoted number
peak_tflops_bf16: "800"    # string, not float

# RIGHT
peak_tflops_bf16: 800

# WRONG: missing nested block
calibration: 0.5           # scalar, not a mapping

# RIGHT
calibration:
  compute_eff_large_gemm: 0.50
  compute_eff_small_op: 0.20
  memory_efficiency: 0.70
  comm_efficiency: 0.70
```

---

## `ValueError: num_heads not divisible by tp`

**Symptom:**
```
Error: num_heads (48) must be divisible by tp (8)
```

**Cause:** The tensor parallelism degree does not evenly divide the number of attention heads.

**Fix:** Choose a TP value that divides `num_heads`. Common TP values:
- 48 heads: TP = 1, 2, 3, 4, 6, 8, 12, 16, 24, 48
- 64 heads: TP = 1, 2, 4, 8, 16, 32, 64
- 32 heads: TP = 1, 2, 4, 8, 16, 32

Also check that `num_kv_heads` is divisible by TP for GQA models. For example, if `num_kv_heads=8`, valid TP values are 1, 2, 4, 8.

---

## `NOT FEASIBLE: OOM`

**Symptom:**
```
Epoch time:        3.21 hours  [NOT FEASIBLE: OOM]
Memory:
  Train: 142.3/108.8 GB  [OOM]
  Gen:   72.5/108.8 GB  [OK]
```

**Cause:** The per-device memory requirement exceeds usable HBM.

**Fixes (try in order):**

### Training OOM
1. **Reduce `train_micro_batch_size`** -- Directly reduces activation memory.
2. **Increase TP** -- Shards weights and activations across more devices.
3. **Enable attention recomputation** -- `recompute_attention: true` saves activation memory with only +5% time overhead.
4. **Enable full recomputation** -- `full_recomputation: true` for maximum memory savings (+30% time overhead).
5. **Enable ZeRO stage 1+** -- Shards optimizer states across DP ranks. Set `zero_stage: 1` in parallelism config.
6. **Offload optimizer to CPU** -- `optimizer_offload: true` (+15% time overhead).
7. **Offload reference model** -- Set `ref_offload_cpu: true` in RLConfig.
8. **Increase PP** -- Splits model across pipeline stages, reducing per-device weight and activation memory.

### Generation OOM
1. **Reduce `gen_batch_size`** -- KV cache scales linearly with batch size.
2. **Increase TP** -- Shards KV heads across more devices.
3. **Reduce `max_response_len`** -- KV cache scales with sequence length.

---

## `NOT FEASIBLE: OVER TIME`

**Symptom:**
```
Epoch time:        8.45 hours  [NOT FEASIBLE: OVER TIME]
```

**Cause:** The predicted epoch time exceeds the `--time-budget`.

**Fixes:**
1. **Add more devices** -- Increase DP to reduce the number of batches per epoch.
2. **Increase batch sizes** -- Larger `gen_batch_size` and `train_micro_batch_size` reduce total steps.
3. **Switch to separated mode** -- If `colocated=True`, switching to separate device pools allows generation and training to overlap.
4. **Enable speculative decoding** -- If generation is the bottleneck and the model has MTP heads, set `use_speculative_decoding: true`.
5. **Reduce workload** -- Fewer `total_prompts` or smaller `group_size`.

---

## `NOT FEASIBLE: OOM + OVER TIME`

When both appear together, fix OOM first. Memory-saving techniques (recomputation, offload) add time penalties, so the time budget may change after fixing memory.

---

## Python Version Issues

**Symptom:**
```
ModuleNotFoundError: No module named 'pydantic'
```
or syntax errors in type hints.

**Cause:** rl-perf requires Python 3.10 or later.

**Fix:**
```bash
python3.10 --version   # Verify 3.10+ is available
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## YAML Parsing Errors

**Symptom:**
```
Error: invalid YAML: while parsing a block mapping ...
```

**Cause:** Indentation or syntax error in the YAML file.

**Fixes:**
1. YAML uses spaces, not tabs. Make sure your editor is configured for spaces.
2. Check that nested blocks (like `default_layer`, `calibration`) are indented consistently.
3. Use a YAML linter or validator to check your file.

---

## Unexpected Results

If the predictions seem wrong:

1. **Check calibration values** -- Conservative defaults (50% compute efficiency) may underestimate performance. See [calibration-guide.md](calibration-guide.md).
2. **Check `hbm_usable_ratio`** -- The default 0.85 may not match your setup. Measure actual available memory.
3. **Verify parallelism config** -- Make sure TP, PP, DP match your intended setup. `total_devices = tp * pp * dp * ep`.
4. **Check `colocated` flag** -- `colocated=True` means serial execution (gen + train). `colocated=False` means overlapped (max of gen, train).
5. **Compare with `--format json`** -- JSON output includes all fields for detailed inspection.

```bash
rl-perf targets -m configs/models/qwen2_5_72b.yaml -hw 910C -d 64 -p 10000 -f json
```
