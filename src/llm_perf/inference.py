"""Inference / generation time modeling.

Provides generation_time() for prefill + decode performance estimation,
and effective_response_len() for batch-aware response length calculation.
These can be used directly for inference performance modeling without
depending on RL orchestration logic (ref_time, step_time, reshard).
"""

import math

from llm_perf import ops
from llm_perf.builder import (
    build_generation_step,
)
from llm_perf.config import Phase
from llm_perf._stage_utils import _simulate_slowest_stage


def effective_response_len(
    avg: int, std: int = None, batch_size: int = 1, max_len: int = None
) -> float:
    """Gumbel approximation for expected max of batch_size samples.
    E[max of B samples] ≈ avg + std * sqrt(2 * ln(B))
    Falls back to max_len if std not provided."""
    if std is not None and std > 0 and batch_size > 1:
        return avg + std * math.sqrt(2 * math.log(batch_size))
    if max_len is not None:
        return max_len
    return avg


def _build_prefill_ops(model_cfg, hw, parallel_cfg, rl_cfg, stage_layers=None):
    prefill_ops, _ = build_generation_step(
        model_cfg, hw, parallel_cfg, rl_cfg, stage_layers=stage_layers
    )
    return prefill_ops


def _build_decode_ops(model_cfg, hw, parallel_cfg, rl_cfg, stage_layers=None):
    _, decode_ops = build_generation_step(
        model_cfg, hw, parallel_cfg, rl_cfg, stage_layers=stage_layers
    )
    return decode_ops


def prefill_decode_times(model_cfg, hw, parallel_cfg, rl_cfg):
    """Slowest-stage prefill time and per-token decode time.

    For multi-stage PP, the slowest stage determines the pipeline time.
    Each stage is simulated independently (build_fn rebuilt per stage's own
    layer composition) and the maximum wall-clock time across stages is
    used — same convention as training_time()/ref_time() via
    _simulate_slowest_stage(). This matters for mixed-layer models where
    stage 0 (which _split_stages gives any remainder layers to) is not
    necessarily the slowest stage.

    Returns (t_prefill, prefill_sim, t_decode_per_token).
    """
    t_prefill, prefill_sim = _simulate_slowest_stage(
        _build_prefill_ops, model_cfg, hw, parallel_cfg, rl_cfg
    )
    t_decode_per_token, _ = _simulate_slowest_stage(
        _build_decode_ops, model_cfg, hw, parallel_cfg, rl_cfg
    )
    return t_prefill, prefill_sim, t_decode_per_token


def generation_time(model_cfg, hw, parallel_cfg, rl_cfg):
    """Total generation time in seconds. Returns (prefill_sim, t_step).

    GRPO group-aware generation:
      group_size=16 means each prompt is sampled 16 times.
      A gen_batch of 64 samples covers 64/16=4 distinct prompts.
      Prefill is paid once per prompt (4 times), not per response (64 times).
      Decode is paid per response (64 times × eff_len tokens).

    For multi-stage PP, the slowest stage determines the pipeline time;
    see prefill_decode_times().
    """

    t_prefill, prefill_sim, t_decode_per_token = prefill_decode_times(
        model_cfg, hw, parallel_cfg, rl_cfg
    )

    eff_len = effective_response_len(
        avg=rl_cfg.avg_response_len,
        std=rl_cfg.std_response_len,
        batch_size=rl_cfg.gen_batch_size,
        max_len=rl_cfg.max_response_len,
    )

    t_step = t_prefill + eff_len * t_decode_per_token

    # Speculative decoding throughput multiplier (spec §4.3)
    if rl_cfg.use_speculative_decoding:
        mtp_depth = (model_cfg.auxiliary or {}).get("mtp_depth", 0)
        if mtp_depth > 0:
            acceptance_len = rl_cfg.mtp_acceptance_len or mtp_depth
            draft_cost = ops.op_mtp_head(
                model_cfg.hidden_size,
                model_cfg.vocab_size,
                mtp_depth,
                batch_tokens=rl_cfg.gen_batch_size,
                phase=Phase.DECODE,
                dtype_bytes=model_cfg.dtype_bytes,
            )
            draft_overhead = (
                ops.roofline_time(draft_cost, hw) / t_decode_per_token
                if t_decode_per_token > 0
                else 0
            )
            throughput_multiplier = acceptance_len / (1 + draft_overhead)
            if throughput_multiplier > 0:
                t_step = (
                    t_prefill
                    + (eff_len / throughput_multiplier)
                    * t_decode_per_token
                )

    return prefill_sim, t_step