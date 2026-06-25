"""Shared utility functions for pipeline stage simulation.

Contains PP P2P time estimation, bubble time calculation, CPU offload
transfer time, optimizer offload time, and the slowest-stage simulation
helper. These are used by inference.py, training.py, and pipeline.py.
"""

from llm_perf.builder import (
    build_forward_pass,
    _split_stages,
)
from llm_perf.simulator import simulate


def _cpu_offload_transfer_time(weight_bytes: float, hw) -> float:
    """Time to transfer weights from CPU to GPU over PCIe/HCCS.

    When model weights are offloaded to CPU memory, they must be transferred
    to GPU before each forward pass. The transfer time is:
        t = weight_bytes / (cpu_gpu_bw * 1e9)

    Args:
        weight_bytes: Total weight bytes to transfer (per device, after TP sharding).
        hw: HardwareConfig with cpu_gpu_bw_gb_s field.

    Returns:
        Transfer time in seconds.
    """
    bw_bytes = hw.cpu_gpu_bw_gb_s * 1e9
    return weight_bytes / bw_bytes if bw_bytes > 0 else 0.0


def _compute_pp_p2p_time(model_cfg, hw, parallel_cfg, batch_size, seq_len):
    """PP bubble P2P time for one pipeline step (fwd + bwd).

    In 1F1B, the pipeline bubble at the start of each step contains (pp-1)
    idle micro-batches. Each idle micro-batch requires 2 P2P transfers per
    stage boundary (1 forward activation + 1 backward gradient).
    Total bubble P2Ps = (pp-1) * 2.

    In steady state, P2P transfers overlap with the next micro-batch's
    compute, so only the bubble P2Ps are on the critical path.
    This function returns the total bubble P2P time for the entire step,
    NOT per micro-batch — callers should NOT multiply by micro-batch count.

    Each P2P incurs both bandwidth cost (size / bw) and a latency cost,
    consistent with ops.comm_time's P2P modeling.

    Note: CP shards the sequence across CP ranks, so the activation tensor
    between PP stages is CP-local: local_seq_len = seq_len // cp.
    """
    pp = parallel_cfg.pp
    if pp <= 1:
        return 0.0
    tp = parallel_cfg.tp
    cp = parallel_cfg.cp
    # CP shards the sequence; each PP P2P sends only the local shard
    local_seq_len = seq_len // cp if cp > 1 else seq_len
    # Activation tensor between stages (TP-sharded, each rank sends its shard)
    activation_bytes = batch_size * local_seq_len * model_cfg.hidden_size * model_cfg.dtype_bytes
    activation_bytes_per_tp = activation_bytes // tp
    bw_bytes = hw.inter_node_bw_gb_s * 1e9 * hw.calibration.comm_efficiency
    lat = hw.inter_node_latency_us * 1e-6
    num_p2p = (pp - 1) * 2
    t_p2p = activation_bytes_per_tp / bw_bytes + lat
    return num_p2p * t_p2p


def _compute_pp_p2p_time_fwd(model_cfg, hw, parallel_cfg, batch_size, seq_len):
    """PP bubble P2P time for one pipeline step (forward only, no backward).

    Same as _compute_pp_p2p_time but for forward-only passes (e.g. reference,
    reward, old_logprob). Only (pp-1) forward P2Ps are on the critical path.
    This is the total bubble P2P time for the step — callers should NOT
    multiply by micro-batch count.

    Each P2P incurs both bandwidth cost and latency, consistent with
    ops.comm_time's P2P modeling.

    Note: CP shards the sequence across CP ranks, so the activation tensor
    between PP stages is CP-local: local_seq_len = seq_len // cp.
    """
    pp = parallel_cfg.pp
    if pp <= 1:
        return 0.0
    tp = parallel_cfg.tp
    cp = parallel_cfg.cp
    # CP shards the sequence; each PP P2P sends only the local shard
    local_seq_len = seq_len // cp if cp > 1 else seq_len
    activation_bytes = batch_size * local_seq_len * model_cfg.hidden_size * model_cfg.dtype_bytes
    activation_bytes_per_tp = activation_bytes // tp
    bw_bytes = hw.inter_node_bw_gb_s * 1e9 * hw.calibration.comm_efficiency
    lat = hw.inter_node_latency_us * 1e-6
    num_p2p = pp - 1  # forward only
    t_p2p = activation_bytes_per_tp / bw_bytes + lat
    return num_p2p * t_p2p


def _pp_bubble_time(t_non_bubble, parallel_cfg, num_micro_batches):
    """PP bubble idle time for a pipeline execution.

    In 1F1B, the pipeline warmup/cooldown leaves (pp-1) micro-batches
    of idle time on some stages. Each micro-batch takes
    t_non_bubble / M time (compute + recompute + p2p), so the bubble
    is (pp-1) * (t_non_bubble / M) = t_non_bubble * (pp-1) / M.

    We express this as:
      bubble_ratio = (pp-1) / (M + pp-1)
      t_bubble = t_non_bubble * bubble_ratio

    This avoids the circular dependency: t_bubble depends on
    t_non_bubble (which excludes the bubble itself), and
    t_step = t_non_bubble + t_bubble.

    Args:
      t_non_bubble: Total non-bubble time (compute + recompute + p2p).
      parallel_cfg: ParallelismConfig with pp field.
      num_micro_batches: Number of micro-batches in this pipeline execution.

    Returns:
      Bubble idle time in seconds.
     """
    if parallel_cfg.pp <= 1 or num_micro_batches <= 0:
            return 0.0
    M = num_micro_batches
    bubble_ratio = (parallel_cfg.pp - 1) / (M + parallel_cfg.pp - 1)
    return t_non_bubble * bubble_ratio


def _optimizer_offload_time(model_cfg, hw, weight_bytes):
    """CPU offload transfer time for optimizer states.

    ZeRO-Offload: gradients (GPU→CPU, bf16) + updated master weights (CPU→GPU, fp32).
    Total = param_count * (2 + 4) = param_count * 6 bytes.
    weight_bytes is per-device (after TP sharding).
    """
    param_count = weight_bytes / model_cfg.dtype_bytes
    grad_bytes = param_count * 2    # bf16 gradients GPU→CPU
    master_bytes = param_count * 4  # fp32 master weights CPU→GPU
    total_bytes = grad_bytes + master_bytes
    return total_bytes / (hw.cpu_gpu_bw_gb_s * 1e9)


def _simulate_slowest_stage(build_fn, model_cfg, hw, parallel_cfg, rl_cfg, **kwargs):
    """Simulate all PP stages and return (slowest_time, stage0_sim_result).

    For mixed-layer models (e.g. 4 SwiGLU + 47 MoE), different PP stages
    have different layer compositions and thus different compute times.
    The pipeline time is determined by the slowest stage.

    Returns (max_wall_clock_time, stage0_sim_result) where stage0_sim_result
    provides weight_bytes and peak_activation_bytes for memory analysis.

    Args:
        build_fn: Builder function (build_training_step, build_forward_pass, etc.)
        model_cfg: ModelConfig.
        hw: HardwareConfig.
        parallel_cfg: ParallelismConfig.
        rl_cfg: WorkloadConfig.
        **kwargs: Additional keyword arguments passed to build_fn.

    Returns:
        Tuple of (slowest_wall_clock_time, stage0_sim_result).
    """
    pp = parallel_cfg.pp
    if pp <= 1:
        ops_list = build_fn(model_cfg, hw, parallel_cfg, rl_cfg, **kwargs)
        sim = simulate(ops_list)
        return sim.wall_clock_time, sim

    all_layers = model_cfg.get_layers()
    stages = _split_stages(all_layers, pp)

    slowest_time = 0.0
    stage0_sim = None

    for i, stage_layers in enumerate(stages):
        ops_list = build_fn(
            model_cfg, hw, parallel_cfg, rl_cfg,
            stage_layers=stage_layers, **kwargs,
        )
        sim = simulate(ops_list)
        stage_time = sim.wall_clock_time
        if i == 0:
            stage0_sim = sim
        if stage_time > slowest_time:
            slowest_time = stage_time

    return slowest_time, stage0_sim