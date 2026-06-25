"""Training time modeling — generic fwd+bwd+optimizer for pretraining.

Provides pretraining_time() for a single training step's performance
estimation: forward, backward, optimizer, recompute, PP P2P, PP bubble,
and optimizer offload. No RL-specific sub-steps
(reward_fwd, old_logprob_fwd) are included — those belong in
post_training.rl_training_time().
"""

from llm_perf.builder import build_training_step
from llm_perf.report import TrainBreakdown
from llm_perf._stage_utils import (
    _simulate_slowest_stage,
    _compute_pp_p2p_time,
    _pp_bubble_time,
    _optimizer_offload_time,
)


def pretraining_time(model_cfg, hw, parallel_cfg, rl_cfg):
    """Single training step time (fwd+bwd+optimizer). Returns (t_step, sim_result, TrainBreakdown).

    Models a generic training step:
      1. LLM forward+backward (build_training_step)
      2. Optimizer step (with optional CPU offload)
      3. Recomputation overhead
      4. PP P2P communication
      5. PP bubble idle time

    The rl_cfg parameter provides training-hyperparameter fields
    (train_batch_size, train_micro_batch_size, gradient_accumulation_steps,
    avg_prompt_len, avg_response_len) that are also meaningful for
    pretraining. RL-specific fields (reward_model, group_size, etc.)
    are ignored by this function.

    Returns:
        (t_step, sim_result, TrainBreakdown) where TrainBreakdown.reward_fwd
        and old_logprob_fwd are always 0.
    """

    # --- 1. LLM forward+backward ---
    t_policy_update, train_sim = _simulate_slowest_stage(
        build_training_step, model_cfg, hw, parallel_cfg, rl_cfg
    )

    # --- 2. Optimizer offload ---
    t_optim_offload_policy = 0.0
    if parallel_cfg.optimizer_offload:
        t_optim_offload_policy = _optimizer_offload_time(model_cfg, hw, train_sim.weight_bytes)

    # --- Per-step total ---
    t_policy_update *= rl_cfg.train_batch_size / parallel_cfg.dp
    t_optim_offload_policy *= rl_cfg.train_batch_size / parallel_cfg.dp

    # Compute total (before bubble and recompute overheads)
    t_compute = t_policy_update + t_optim_offload_policy

    # Recomputation overhead
    recompute_penalty = 1.0
    if parallel_cfg.full_recomputation:
        recompute_penalty *= 1.33
    elif parallel_cfg.recompute_attention:
        recompute_penalty *= 1.05
    if parallel_cfg.activation_offload:
        recompute_penalty *= 1.10
    t_recompute = t_compute * (recompute_penalty - 1.0)

    # PP bubble overhead
    t_non_bubble = t_compute + t_recompute
    M_train = (rl_cfg.train_batch_size
                / rl_cfg.gradient_accumulation_steps
                / rl_cfg.train_micro_batch_size
                / parallel_cfg.dp)
    t_bubble = _pp_bubble_time(t_non_bubble, parallel_cfg, M_train)

    t_step = t_non_bubble + t_bubble

    step_bd = TrainBreakdown(
        reward_fwd=0.0,
        old_logprob_fwd=0.0,
        policy_update=t_policy_update,
        pp_bubble=t_bubble,
        recompute=t_recompute,
        optim_offload=t_optim_offload_policy,
        total=t_step,
    )

    return t_step, train_sim, step_bd