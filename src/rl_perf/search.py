"""Pareto search and sensitivity sweep engine for RL performance modeling.

This module provides pure-logic search utilities that enumerate parallelism
configurations, run performance predictions, and identify the Pareto frontier
for device-count vs step-time trade-offs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from rl_perf.config import HardwareConfig, ParallelismConfig, RLConfig
from rl_perf.model import RLPerformanceModel
from rl_perf.report import TargetReport


@dataclass
class SearchResult:
    """Single configuration evaluation result.

    Attributes:
        devices: Total number of devices used.
        gen_parallel: Parallelism config used for the generation phase.
        train_parallel: Parallelism config used for the training phase.
        ref_parallel: Parallelism config used for the reference phase.
        report: Full TargetReport from derive_targets().
        is_feasible: True if memory fits in HBM.
        is_oom: True if any memory phase is OOM.
        is_pareto: True if this point lies on the Pareto frontier
            (devices vs step_time_seconds). Defaults to False.
    """

    devices: int
    gen_parallel: ParallelismConfig
    train_parallel: ParallelismConfig
    ref_parallel: ParallelismConfig
    report: TargetReport
    is_feasible: bool
    is_oom: bool
    is_pareto: bool = False


def _is_moe_model(model: RLPerformanceModel) -> bool:
    """Return True if any layer in the model uses MoE FFN."""
    for layer in model.model.get_layers():
        if layer.ffn == "MoE":
            return True
    return False


def _enumerate_parallelism(
    device_count: int,
    devices_per_node: int,
    num_layers: int,
    is_moe: bool,
) -> List[tuple[int, int, int, int, int]]:
    """Enumerate valid (tp, pp, ep, dp, cp) combinations for a given device count.

    Rules:
    - tp in [1, 2, 4, 8], tp <= devices_per_node
    - pp in [1, 2, 4, 8], pp <= num_layers (uneven layer distribution is supported)
    - ep in [1] for dense, [1, 2, 4, 8] for MoE
    - cp in [1, 2, 4] for training (cp shards the sequence across cp ranks)
    - dp = device_count // (tp * pp * ep), must be >= 1
    - device_count % (tp * pp * ep) == 0
    """
    valid_tp = [v for v in [1, 2, 4, 8] if v <= devices_per_node]
    valid_pp = [v for v in [1, 2, 4, 8] if v <= num_layers]
    valid_ep = [1, 2, 4, 8] if is_moe else [1]
    valid_cp = [1, 2, 4]

    combos = []
    for tp in valid_tp:
        for pp in valid_pp:
            for ep in valid_ep:
                world = tp * pp * ep
                if device_count % world != 0:
                    continue
                dp = device_count // world
                if dp < 1:
                    continue
                for cp in valid_cp:
                    combos.append((tp, pp, ep, dp, cp))
    return combos


def pareto_search(
    model: RLPerformanceModel,
    hw: HardwareConfig,
    rl_cfg: RLConfig,
    device_counts: List[int],
) -> List[SearchResult]:
    """Enumerate valid TP/PP/EP/DP/CP combos for each device count, run prediction,
    and mark the Pareto frontier (fewer devices AND lower step time).

    Args:
        model: RLPerformanceModel instance.
        hw: HardwareConfig for the target hardware.
        rl_cfg: RLConfig describing the RL workload.
        device_counts: List of total device counts to evaluate.

    Returns:
        List of SearchResult, one per valid configuration evaluated.
        Pareto-optimal feasible points have is_pareto=True.
    """
    is_moe = _is_moe_model(model)
    num_layers = model.model.num_layers
    devices_per_node = hw.devices_per_node

    results: List[SearchResult] = []

    for device_count in device_counts:
        combos = _enumerate_parallelism(
            device_count, devices_per_node, num_layers, is_moe
        )
        for tp, pp, ep, dp, cp in combos:
            # Gen: tp=tp, pp=1; split remaining devices across ep and dp
            gen_ep = ep
            gen_remaining = max(1, device_count // tp // gen_ep)
            gen_parallel = ParallelismConfig(tp=tp, pp=1, ep=gen_ep, dp=gen_remaining)
            # Train: full tp/pp/ep/dp/cp as enumerated; enable sp when cp > 1
            sp = cp > 1
            train_parallel = ParallelismConfig(tp=tp, pp=pp, dp=dp, ep=ep, cp=cp, sp=sp)
            # Ref: simple forward pass — pp=1, no ep needed, maximize dp
            ref_dp = max(1, device_count // tp)
            ref_parallel = ParallelismConfig(tp=tp, pp=1, dp=ref_dp)

            try:
                report = model.derive_targets(
                    device_count,
                    rl_cfg,
                    gen_parallel,
                    train_parallel,
                    ref_parallel,
                )
            except (ValueError, ZeroDivisionError):
                continue

            is_oom = not (report.memory.train_feasible and report.memory.gen_feasible and report.memory.ref_feasible)
            is_feasible = report.feasible

            results.append(
                SearchResult(
                    devices=device_count,
                    gen_parallel=gen_parallel,
                    train_parallel=train_parallel,
                    ref_parallel=ref_parallel,
                    report=report,
                    is_feasible=is_feasible,
                    is_oom=is_oom,
                )
            )

    # Mark Pareto frontier among feasible results only.
    # A feasible point is Pareto if no other feasible point strictly dominates it
    # on both axes: fewer (or equal) devices AND lower (or equal) step time,
    # with at least one strict improvement.
    feasible = [r for r in results if r.is_feasible]
    for candidate in feasible:
        dominated = any(
            other.devices <= candidate.devices
            and other.report.step_time_seconds <= candidate.report.step_time_seconds
            and (
                other.devices < candidate.devices
                or other.report.step_time_seconds < candidate.report.step_time_seconds
            )
            for other in feasible
            if other is not candidate
        )
        if not dominated:
            candidate.is_pareto = True

    return results


def sensitivity_sweep(
    model: RLPerformanceModel,
    hw: HardwareConfig,
    rl_cfg: RLConfig,
    param_name: str,
    values: List,
    total_devices: int,
    gen_parallel: ParallelismConfig,
    train_parallel: ParallelismConfig,
    ref_parallel: Optional[ParallelismConfig] = None,
) -> List[SearchResult]:
    """Sweep a single RLConfig parameter across a list of values and collect results.

    For each value, creates a modified rl_cfg via model_copy(update={param_name: value})
    and calls derive_targets() with the provided parallelism configs.

    Args:
        model: RLPerformanceModel instance.
        hw: HardwareConfig (kept for API symmetry; not used directly here).
        rl_cfg: Base RLConfig to modify.
        param_name: Name of the RLConfig field to sweep.
        values: Ordered list of values to sweep over.
        total_devices: Total device count to pass to derive_targets().
        gen_parallel: ParallelismConfig for the generation phase.
        train_parallel: ParallelismConfig for the training phase.
        ref_parallel: ParallelismConfig for the reference phase.
            If None, defaults to a simple config with pp=1 and max dp.

    Returns:
        List of SearchResult, one per value in the same order as values.
    """
    if ref_parallel is None:
        ref_dp = max(1, total_devices // train_parallel.tp)
        ref_parallel = ParallelismConfig(tp=train_parallel.tp, pp=1, dp=ref_dp)

    results: List[SearchResult] = []

    for val in values:
        modified_cfg = rl_cfg.model_copy(update={param_name: val})
        try:
            report = model.derive_targets(
                total_devices,
                modified_cfg,
                gen_parallel,
                train_parallel,
                ref_parallel,
            )
        except (ValueError, ZeroDivisionError):
            raise

        is_oom = not (report.memory.train_feasible and report.memory.gen_feasible and report.memory.ref_feasible)
        is_feasible = report.feasible

        results.append(
            SearchResult(
                devices=total_devices,
                gen_parallel=gen_parallel,
                ref_parallel=ref_parallel,
                train_parallel=train_parallel,
                report=report,
                is_feasible=is_feasible,
                is_oom=is_oom,
            )
        )

    return results
