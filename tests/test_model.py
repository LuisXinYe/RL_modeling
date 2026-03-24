"""Tests for model.py (RLPerformanceModel) and report.py (format_table)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rl_perf.config import (
    HardwareConfig,
    ModelConfig,
    ParallelismConfig,
    RLConfig,
    load_hardware_config,
    load_model_config,
)
from rl_perf.model import RLPerformanceModel
from rl_perf.report import TargetReport, format_table

CONFIGS_DIR = Path(__file__).parent.parent / "configs"

VALID_BOTTLENECKS = {"GENERATION", "TRAINING", "BALANCED"}


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
def rl_cfg() -> RLConfig:
    return RLConfig(
        total_prompts=64,
        group_size=4,
        avg_prompt_len=256,
        avg_response_len=512,
        max_response_len=1024,
        train_micro_batch_size=2,
        gradient_accumulation_steps=1,
        gen_batch_size=16,
    )


@pytest.fixture
def perf_model(model_cfg, hw) -> RLPerformanceModel:
    return RLPerformanceModel(model_cfg, hw)


# ---------------------------------------------------------------------------
# Test 1: derive_targets — basic sanity
# ---------------------------------------------------------------------------


def test_derive_targets(perf_model, rl_cfg, parallel_cfg):
    report = perf_model.derive_targets(
        total_devices=64,
        rl_cfg=rl_cfg,
        gen_parallel=parallel_cfg,
        train_parallel=parallel_cfg,
        time_budget_hours=10.0,
    )
    assert isinstance(report, TargetReport)
    assert report.epoch_time_hours > 0
    assert report.gen_tps_target > 0
    assert report.train_tps_target > 0
    assert report.bottleneck in VALID_BOTTLENECKS
    # within_budget is a bool
    assert isinstance(report.within_budget, bool)


# ---------------------------------------------------------------------------
# Test 2: feasibility_check — no time budget, within_budget always True
# ---------------------------------------------------------------------------


def test_feasibility_check(perf_model, rl_cfg, parallel_cfg):
    report = perf_model.feasibility_check(
        total_devices=64,
        rl_cfg=rl_cfg,
        gen_parallel=parallel_cfg,
        train_parallel=parallel_cfg,
    )
    assert isinstance(report, TargetReport)
    assert report.epoch_time_hours > 0
    assert report.within_budget is True  # no budget provided → always feasible


# ---------------------------------------------------------------------------
# Test 3: what-if group_size — larger group should take longer training time
# ---------------------------------------------------------------------------


def test_what_if_group_size(perf_model, parallel_cfg):
    rl_small = RLConfig(
        total_prompts=64,
        group_size=8,
        avg_prompt_len=256,
        avg_response_len=512,
        max_response_len=1024,
        train_micro_batch_size=2,
        gen_batch_size=16,
    )
    rl_large = RLConfig(
        total_prompts=64,
        group_size=16,
        avg_prompt_len=256,
        avg_response_len=512,
        max_response_len=1024,
        train_micro_batch_size=2,
        gen_batch_size=16,
    )
    report_small = perf_model.derive_targets(64, rl_small, parallel_cfg, parallel_cfg)
    report_large = perf_model.derive_targets(64, rl_large, parallel_cfg, parallel_cfg)

    # group_size=16 → 2x total_responses → should take longer overall
    assert report_large.epoch_time_hours > report_small.epoch_time_hours


# ---------------------------------------------------------------------------
# Test 4: memory profile — all numeric fields positive, booleans exist
# ---------------------------------------------------------------------------


def test_memory_profile(perf_model, rl_cfg, parallel_cfg):
    report = perf_model.feasibility_check(
        total_devices=64,
        rl_cfg=rl_cfg,
        gen_parallel=parallel_cfg,
        train_parallel=parallel_cfg,
    )
    mem = report.memory
    assert mem.weight_gb > 0
    assert mem.optimizer_gb > 0
    assert mem.activation_peak_gb > 0
    assert mem.kv_cache_gb > 0
    assert mem.total_train_gb > 0
    assert mem.total_gen_gb > 0
    assert mem.usable_hbm_gb > 0
    assert isinstance(mem.train_feasible, bool)
    assert isinstance(mem.gen_feasible, bool)


# ---------------------------------------------------------------------------
# Test 5: format_table returns string with key words
# ---------------------------------------------------------------------------


def test_format_table(perf_model, rl_cfg, parallel_cfg):
    report = perf_model.feasibility_check(
        total_devices=64,
        rl_cfg=rl_cfg,
        gen_parallel=parallel_cfg,
        train_parallel=parallel_cfg,
    )
    table = format_table(report)
    assert isinstance(table, str)
    assert "RL Training Performance Report" in table
    assert "Bottleneck" in table
    assert "Generation" in table
    assert "Training" in table
    assert "Memory" in table
    # Should contain either OK or OOM
    assert "OK" in table or "OOM" in table
