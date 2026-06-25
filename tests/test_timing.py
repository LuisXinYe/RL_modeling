"""Tests for the timing primitives split out of the old pipeline.py:

- llm_perf.inference: generation_time, effective_response_len
- llm_perf.post_training: training_time (RL step), step_time (wall-clock)

(The monolithic pipeline.py was split into inference/training/post_training;
its epoch_time/bottleneck_analysis helpers were removed in favour of
step_time and the per-phase TargetReport fields.)
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from llm_perf.config import (
    HardwareConfig,
    ModelConfig,
    ParallelismConfig,
    WorkloadConfig,
    load_hardware_config,
    load_model_config,
)
from llm_perf.inference import effective_response_len, generation_time
from llm_perf.post_training import step_time, training_time
from llm_perf.simulator import SimResult

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


@pytest.fixture
def model_cfg() -> ModelConfig:
    return load_model_config(str(CONFIGS_DIR / "models" / "llama3_1_8b.yaml"))


@pytest.fixture
def hw() -> HardwareConfig:
    return load_hardware_config(str(CONFIGS_DIR / "hardware" / "ascend_910c.yaml"))


@pytest.fixture
def parallel_cfg() -> ParallelismConfig:
    return ParallelismConfig(tp=8, pp=1, dp=8, ep=1)


@pytest.fixture
def rl_cfg() -> WorkloadConfig:
    return WorkloadConfig(
        group_size=4,
        avg_prompt_len=256,
        avg_response_len=512,
        max_response_len=1024,
        std_response_len=200,
        train_micro_batch_size=2,
        gradient_accumulation_steps=1,
        gen_batch_size=16,
    )


# ---------------------------------------------------------------------------
# effective_response_len
# ---------------------------------------------------------------------------


def test_effective_response_len_with_std():
    avg, std, batch = 2048, 800, 32
    result = effective_response_len(avg=avg, std=std, batch_size=batch)
    # Gumbel approx: avg + std * sqrt(2 * ln(B))
    expected = avg + std * math.sqrt(2 * math.log(batch))
    assert result > avg
    assert result == pytest.approx(expected, rel=1e-6)
    assert result < avg + std * 10


def test_effective_response_len_fallback_max():
    result = effective_response_len(avg=2048, std=None, batch_size=32, max_len=4096)
    assert result == 4096


def test_effective_response_len_no_std_no_max():
    result = effective_response_len(avg=512)
    assert result == 512


# ---------------------------------------------------------------------------
# step_time (replaces the old epoch_time helper)
# ---------------------------------------------------------------------------


def test_step_time_serial():
    # Phases run serially on the shared pool: gen + ref + train.
    t = step_time(t_gen=10.0, t_train=15.0, t_ref=0.0)
    assert t == pytest.approx(25.0)


def test_step_time_with_reshard():
    t = step_time(t_gen=10.0, t_train=15.0, t_ref=2.0, t_reshard=1.0)
    assert t == pytest.approx(28.0)  # 10 + 2 + 15 + 1


# ---------------------------------------------------------------------------
# generation_time — returns (SimResult, t_step)
# ---------------------------------------------------------------------------


def test_generation_time_nonzero(model_cfg, hw, parallel_cfg, rl_cfg):
    _, t_step = generation_time(model_cfg, hw, parallel_cfg, rl_cfg)
    assert t_step > 0


def test_generation_time_returns_tuple(model_cfg, hw, parallel_cfg, rl_cfg):
    """generation_time returns (prefill_sim, t_step)."""
    result = generation_time(model_cfg, hw, parallel_cfg, rl_cfg)
    assert isinstance(result, tuple)
    assert len(result) == 2
    sim, t_step = result
    assert isinstance(sim, SimResult)
    assert t_step > 0


def test_generation_time_includes_decode(model_cfg, hw, parallel_cfg, rl_cfg):
    """Total gen time should exceed prefill-only time (decode is included)."""
    from llm_perf.builder import build_generation_step
    from llm_perf.simulator import simulate as sim_fn

    _, t_step = generation_time(model_cfg, hw, parallel_cfg, rl_cfg)
    prefill_ops, _ = build_generation_step(model_cfg, hw, parallel_cfg, rl_cfg)
    t_prefill = sim_fn(prefill_ops).wall_clock_time
    assert t_step > t_prefill


def test_gen_dp_reduces_time_via_kv_read(model_cfg, hw):
    """Larger gen DP → smaller per-card batch → less decode KV traffic → shorter
    generation. Regression guard that decode models KV-cache reads (so generation
    parallelism actually matters)."""
    rl = WorkloadConfig(
        group_size=8, gen_batch_size=256,
        avg_prompt_len=512, avg_response_len=2048, max_response_len=2048,
    )
    _, t_dp1 = generation_time(model_cfg, hw, ParallelismConfig(tp=4, pp=1, dp=1), rl)
    _, t_dp8 = generation_time(model_cfg, hw, ParallelismConfig(tp=4, pp=1, dp=8), rl)
    assert t_dp8 < t_dp1 * 0.7


def test_speculative_decoding_reduces_gen_time():
    """Speculative decoding with acceptance_len > 1 should reduce generation time."""
    mc = load_model_config(str(CONFIGS_DIR / "models" / "deepseekv3_671b.yaml"))
    hw = load_hardware_config(str(CONFIGS_DIR / "hardware" / "ascend_910c.yaml"))
    parallel = ParallelismConfig(tp=8, pp=1, dp=8)

    rl_base = WorkloadConfig(group_size=4, gen_batch_size=16, train_micro_batch_size=2)
    rl_spec = WorkloadConfig(
        group_size=4, gen_batch_size=16, train_micro_batch_size=2,
        use_speculative_decoding=True, mtp_acceptance_len=2,
    )

    _, t_base = generation_time(mc, hw, parallel, rl_base)
    _, t_spec = generation_time(mc, hw, parallel, rl_spec)
    assert t_spec < t_base


# ---------------------------------------------------------------------------
# training_time — returns (t_step, SimResult, TrainBreakdown)
# ---------------------------------------------------------------------------


def test_training_time_nonzero(model_cfg, hw, parallel_cfg, rl_cfg):
    t, _, _ = training_time(model_cfg, hw, parallel_cfg, rl_cfg)
    assert t > 0


def test_training_time_returns_tuple(model_cfg, hw, parallel_cfg, rl_cfg):
    """training_time returns (t_step, SimResult, TrainBreakdown)."""
    result = training_time(model_cfg, hw, parallel_cfg, rl_cfg)
    assert isinstance(result, tuple)
    assert len(result) == 3
    t, sim, bd = result
    assert t > 0
    assert isinstance(sim, SimResult)
    assert sim.weight_bytes > 0
    assert bd.total > 0


def test_pp_bubble_ratio(model_cfg, hw, rl_cfg):
    """PP > 1 should change training step time (bubble overhead vs fewer layers/stage)."""
    parallel_pp1 = ParallelismConfig(tp=8, pp=1, dp=8)
    parallel_pp4 = ParallelismConfig(tp=8, pp=4, dp=2)

    t1, _, _ = training_time(model_cfg, hw, parallel_pp1, rl_cfg)
    t4, _, _ = training_time(model_cfg, hw, parallel_pp4, rl_cfg)
    assert t1 > 0
    assert t4 > 0
    assert t1 != t4
