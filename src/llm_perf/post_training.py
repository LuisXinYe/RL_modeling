"""RL post-training orchestration — RL training, reference, step time, resharding.

Re-exports generation_time and pretraining_time from their dedicated modules
for convenient access from a single import.
"""

from llm_perf.builder import build_forward_pass
from llm_perf._stage_utils import (
    _simulate_slowest_stage,
    _compute_pp_p2p_time_fwd,
    _cpu_offload_transfer_time,
    _pp_bubble_time,
)
from llm_perf.inference import generation_time               # re-export
from llm_perf.training import pretraining_time               # re-export
from llm_perf.report import TrainBreakdown


def rl_training_time(model_cfg, hw, parallel_cfg, rl_cfg):
    """RL training step time (reward_fwd + old_logprob_fwd + policy_update).

    Returns (total_time, sim_result, TrainBreakdown).

    Each step of the RL training phase:
      1. reward:        reward model forward (if reward_model)
      2. old_log_prob:  policy forward to compute old log probs
      3. update_actor:  policy fwd+bwd update (via pretraining_time)

    Calls pretraining_time() for the generic fwd+bwd+optimizer core,
    then adds RL-specific forward passes and recalculates overheads.

    Note: advantage computation (GRPO group normalization) is
    O(B×S) element-wise, negligible vs forward passes — not modeled.
    """

    # --- 1. Reward model forward (if enabled) ---
    t_reward_fwd = 0.0
    t_pp_p2p_reward = 0.0
    if rl_cfg.reward_model:
        t_reward_fwd, _ = _simulate_slowest_stage(
            build_forward_pass, model_cfg, hw, parallel_cfg, rl_cfg, name_prefix="reward_"
        )
        t_pp_p2p_reward = _compute_pp_p2p_time_fwd(
            model_cfg, hw, parallel_cfg,
            rl_cfg.train_micro_batch_size,
            rl_cfg.avg_prompt_len + rl_cfg.avg_response_len,
        )

    # --- 2. old_log_prob forward (GRPO) ---
    t_old_fwd, _ = _simulate_slowest_stage(
        build_forward_pass, model_cfg, hw, parallel_cfg, rl_cfg, name_prefix="old_"
    )
    t_pp_p2p_old = _compute_pp_p2p_time_fwd(
        model_cfg, hw, parallel_cfg,
        rl_cfg.train_micro_batch_size,
        rl_cfg.avg_prompt_len + rl_cfg.avg_response_len,
    )

    # --- 3. Policy update: delegate to pretraining_time ---
    _, train_sim, pretrain_bd = pretraining_time(model_cfg, hw, parallel_cfg, rl_cfg)

    # Scale RL-specific forward passes by number of micro-batches
    t_reward_fwd *= rl_cfg.train_batch_size / parallel_cfg.dp
    t_old_fwd *= rl_cfg.train_batch_size / parallel_cfg.dp

    # Extra PP P2P from reward and old_logprob forwards
    t_pp_p2p_extra = t_pp_p2p_reward + t_pp_p2p_old

    # Recalculate compute total including RL-specific forwards
    t_compute = (
        t_reward_fwd
        + t_old_fwd
        + pretrain_bd.policy_update
        + pretrain_bd.optim_offload
    )

    # Recomputation overhead (same penalty ratios as pretraining)
    recompute_penalty = 1.0
    if parallel_cfg.full_recomputation:
        recompute_penalty *= 1.33
    elif parallel_cfg.recompute_attention:
        recompute_penalty *= 1.05
    if parallel_cfg.activation_offload:
        recompute_penalty *= 1.10
    t_recompute = t_compute * (recompute_penalty - 1.0)

    # PP bubble
    t_non_bubble = t_compute + t_recompute
    M_train = (rl_cfg.train_batch_size
                / rl_cfg.gradient_accumulation_steps
                / rl_cfg.train_micro_batch_size
                / parallel_cfg.dp)
    t_bubble = _pp_bubble_time(t_non_bubble, parallel_cfg, M_train)

    t_step = t_non_bubble + t_bubble

    step_bd = TrainBreakdown(
        reward_fwd=t_reward_fwd,
        old_logprob_fwd=t_old_fwd,
        policy_update=pretrain_bd.policy_update,
        pp_bubble=t_bubble,
        recompute=t_recompute,
        optim_offload=pretrain_bd.optim_offload,
        total=t_step,
    )

    return t_step, train_sim, step_bd


# Backward-compatible alias
training_time = rl_training_time


def ref_time(model_cfg, hw, parallel_cfg, rl_cfg):
    """Total reference time in seconds. Returns (total_time, sim_result).
    """

    # --- 1. Reference model forward (if enabled) ---
    t_ref_fwd = 0.0
    ref_sim = None
    t_pp_p2p_ref = 0.0
    t_ref_offload = 0.0
    if rl_cfg.reference_model:
        t_ref_fwd, ref_sim = _simulate_slowest_stage(
            build_forward_pass, model_cfg, hw, parallel_cfg, rl_cfg, name_prefix="ref_"
        )
        t_pp_p2p_ref = _compute_pp_p2p_time_fwd(model_cfg, hw, parallel_cfg, rl_cfg.train_batch_size * rl_cfg.group_size / parallel_cfg.dp, rl_cfg.avg_prompt_len + rl_cfg.avg_response_len)
        # CPU offload: add weight transfer time (CPU → GPU before forward)
        if rl_cfg.ref_offload_cpu:
            t_ref_offload = _cpu_offload_transfer_time(ref_sim.weight_bytes, hw)

    t_ref_fwd *= rl_cfg.train_batch_size * rl_cfg.group_size / parallel_cfg.dp

    # PP bubble: ref forward runs N micro-batches through the pipeline,
    # incurring one bubble per pipeline execution.
    M_ref = rl_cfg.train_batch_size * rl_cfg.group_size / parallel_cfg.dp
    t_ref_non_bubble = t_ref_fwd + t_pp_p2p_ref + t_ref_offload
    t_ref_bubble = _pp_bubble_time(t_ref_non_bubble, parallel_cfg, M_ref)

    t_ref = t_ref_non_bubble + t_ref_bubble

    return t_ref, ref_sim


def step_time(
    t_gen: float,
    t_train: float,
    t_ref: float,
    startup_overhead: float = 0,
    colocated: bool = False,
    t_reshard: float = 0,
) -> float:
    """Compute wall-clock time.

    Args:
        t_gen: Generation phase time.
        t_train: Training phase time.
        t_ref: Reference phase time.
        startup_overhead: Startup overhead (non-colocated only).
        colocated: If True, phases run sequentially on same devices.
        t_reshard: Resharding time between phases (colocated only).
    """
    if colocated:
        return t_gen + t_ref + t_train + t_reshard
    return max(t_gen, t_train, t_ref) + startup_overhead


def _reshard_time(
    model_cfg,
    hw,
    src_parallel,
    dst_parallel,
) -> float:
    """Estimate time to reshard model weights between two parallelism configs.

    When the parallelism strategy changes between phases (e.g. gen→train,
    train→ref), model weights must be redistributed across devices.
    This involves collective communication (AllGather + ReduceScatter or
    AllToAll) proportional to the total model weight volume.

    Model:
      - The full model weights must be reassembled from the source layout
        and redistributed into the destination layout.
      - Each device holds weight_bytes / (tp_src * pp_src * ep_src) in the
        source layout and needs weight_bytes / (tp_dst * pp_dst * ep_dst)
        in the destination layout.
      - Resharding is modeled as an AllToAll over the device group that
        spans both layouts, with total communication volume ≈ total_weight.
      - For PP changes, each device needs to load/unload different layer
        sets, but the total data movement is still bounded by total_weight.

    Args:
        model_cfg: ModelConfig.
        hw: HardwareConfig.
        src_parallel: Source ParallelismConfig.
        dst_parallel: Destination ParallelismConfig.

    Returns:
        Estimated resharding time in seconds.
    """
    # If parallelism is identical, no resharding needed
    if (src_parallel.tp == dst_parallel.tp
            and src_parallel.pp == dst_parallel.pp
            and src_parallel.ep == dst_parallel.ep
            and src_parallel.dp == dst_parallel.dp):
        return 0.0

    # Estimate total model weight bytes (before any sharding)
    all_layers = model_cfg.get_layers()
    total_weight_bytes = 0.0
    dtype_bytes = model_cfg.dtype_bytes
    d = model_cfg.hidden_size

    for layer_cfg in all_layers:
        # Attention weights
        if layer_cfg.attention in ("GQA", "MHA", "SWA"):
            q_params = d * layer_cfg.num_heads * layer_cfg.head_dim
            kv_params = 2 * d * layer_cfg.num_kv_heads * layer_cfg.head_dim
            o_params = layer_cfg.num_heads * layer_cfg.head_dim * d
            total_weight_bytes += (q_params + kv_params + o_params) * dtype_bytes
        elif layer_cfg.attention == "MLA":
            total_weight_bytes += (
                d * layer_cfg.query_compression_dim
                + layer_cfg.query_compression_dim * d
                + d * layer_cfg.kv_compression_dim
                + layer_cfg.kv_compression_dim * d
                + layer_cfg.kv_compression_dim * d
                + d * d
            ) * dtype_bytes
        elif layer_cfg.attention == "DSA":
            q_params = d * layer_cfg.q_lora_rank + layer_cfg.q_lora_rank * layer_cfg.num_heads * layer_cfg.head_dim
            kv_params = d * layer_cfg.head_dim  # MQA
            o_params = layer_cfg.num_heads * layer_cfg.head_dim * layer_cfg.o_lora_rank + layer_cfg.o_groups * layer_cfg.o_lora_rank * d
            total_weight_bytes += (q_params + kv_params + o_params) * dtype_bytes
            if layer_cfg.compress_ratio == 4:
                total_weight_bytes += layer_cfg.q_lora_rank * layer_cfg.index_n_heads * layer_cfg.index_head_dim * dtype_bytes

        # FFN weights
        if layer_cfg.ffn == "SwiGLU":
            total_weight_bytes += 3 * d * layer_cfg.intermediate_size * dtype_bytes
        elif layer_cfg.ffn == "MoE":
            expert_int = layer_cfg.expert_intermediate_size or layer_cfg.intermediate_size
            total_weight_bytes += (
                layer_cfg.num_experts
                * 3
                * d
                * expert_int
                * dtype_bytes
            )
            if layer_cfg.num_shared_experts > 0:
                shared_int = (
                    layer_cfg.shared_intermediate_size or layer_cfg.intermediate_size
                )
                total_weight_bytes += (
                    layer_cfg.num_shared_experts
                    * 3 * d * shared_int * dtype_bytes
                )

        # RMSNorm
        total_weight_bytes += 2 * d * dtype_bytes

        # mHC weights
        if layer_cfg.residual == "mHC":
            n = layer_cfg.mhc_expansion
            total_weight_bytes += 4 * 3 * n * n * dtype_bytes

    # Add embedding + LM head
    total_weight_bytes += 2 * model_cfg.vocab_size * d * dtype_bytes

    # Add MTP head weight if present
    if model_cfg.auxiliary:
        mtp_depth = model_cfg.auxiliary.get("mtp_depth", 0)
        if mtp_depth > 0:
            total_weight_bytes += mtp_depth * d * model_cfg.vocab_size * dtype_bytes

    # Communication volume: each device sends/receives its shard.
    # Total data movement ≈ total_weight_bytes (each byte moves once).
    # Use the number of devices involved in the resharding group.
    # In colocated mode, the resharding group is the union of src and dst
    # device groups. Since they share the same physical devices, we use
    # the max of the two.
    n_devices = max(src_parallel.total_devices, dst_parallel.total_devices)

    # Determine if communication is intra-node or inter-node
    is_intra = n_devices <= hw.devices_per_node

    if is_intra:
        bw_bytes = hw.intra_node_bw_gb_s * 1e9 * hw.calibration.comm_efficiency
        lat = hw.intra_node_latency_us * 1e-6
    else:
        bw_bytes = hw.inter_node_bw_gb_s * 1e9 * hw.calibration.comm_efficiency
        lat = hw.inter_node_latency_us * 1e-6

    # Model as AllToAll: each device sends total_weight/N to every other device.
    # AllToAll bandwidth cost: per-device volume / bandwidth.
    # Per-device communication volume ≈ total_weight * (N-1)/N ≈ total_weight.
    # Add latency for the AllToAll operation.
    comm_bytes = total_weight_bytes * (n_devices - 1) / n_devices
    t_reshard = comm_bytes / bw_bytes + lat

    return t_reshard