"""Variable-length 1F1B(+V) pipeline simulator over the pool-wide microbatch
abstraction. See docs/superpowers/specs/2026-06-26-dynamic-cp-pipeline-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from llm_perf.builder import SimOp, build_forward_pass
from llm_perf.simulator import simulate


@dataclass
class PoolUnit:
    """One pool-wide pipeline microbatch (fills the whole CP/DP pool).

    cp: context-parallel degree used for this unit.
    seq_len: full sequence length L processed (per-rank tokens = L // cp).
    packed_tokens: total tokens carried by the unit (≈ R·B), for accounting.
    bin_index: source length-bin index (diagnostics / ordering).
    """
    cp: int
    seq_len: int
    packed_tokens: int
    bin_index: int


def stage_unit_time(
    model_cfg, hw, base_par, wl, chunk_layers, chunk_id: int, cp: int,
    seq_len: int, bwd_factor: float = 2.0, cache: Optional[dict] = None,
) -> Tuple[float, float]:
    """Forward/backward time (s) for one pool unit at one virtual stage.

    fwd_t = max(compute, cp_ring_comm) + tp_comm   (CP-ring overlaps compute).
    bwd_t = bwd_factor * fwd_t.
    Cached by (chunk_id, cp, seq_len) — identical bins/stages reuse the sim.
    """
    key = (chunk_id, int(cp), int(seq_len))
    if cache is not None and key in cache:
        return cache[key]
    par = base_par.model_copy(update={"cp": int(cp), "pp": 1, "dp": 1})
    cfg = wl.model_copy(update={
        "avg_prompt_len": int(seq_len), "avg_response_len": 0,
        "train_micro_batch_size": 1,
    })
    sim = simulate(build_forward_pass(model_cfg, hw, par, cfg, stage_layers=chunk_layers))
    fwd_t = max(sim.compute_time, sim.cp_comm_time) + sim.tp_comm_time
    bwd_t = bwd_factor * fwd_t
    result = (fwd_t, bwd_t)
    if cache is not None:
        cache[key] = result
    return result


def _stage_1f1b_order(stage_idx: int, num_stages: int, m: int) -> List[Tuple[int, str]]:
    """Standard 1F1B event order for one stage: (microbatch_idx, phase)."""
    warmup = num_stages - 1 - stage_idx
    warmup = max(0, min(warmup, m))
    events: List[Tuple[int, str]] = []
    f_idx = 0
    b_idx = 0
    # Warmup phase: only forwards
    for _ in range(warmup):
        events.append((f_idx, "F"))
        f_idx += 1
    # 1F1B phase: interleave F and B, one F then one B
    while f_idx < m:
        events.append((f_idx, "F"))
        f_idx += 1
        events.append((b_idx, "B"))
        b_idx += 1
    # Cooldown phase: drain remaining backwards
    while b_idx < m:
        events.append((b_idx, "B"))
        b_idx += 1
    return events


def pipeline_schedule(m: int, p: int, v: int = 1) -> List[List[Tuple[int, int, str]]]:
    """Per-device ordered events for 1F1B(+V interleaved).

    S = p*v virtual stages; vstage s lives on device s % p. For each device,
    round-robin-interleave the standard-1F1B event lists of its v virtual stages.
    """
    S = p * v
    # standard 1F1B order per virtual stage: list of (unit_idx, phase)
    vstage_orders = [_stage_1f1b_order(s, S, m) for s in range(S)]
    per_device: List[List[Tuple[int, int, str]]] = [[] for _ in range(p)]
    for d in range(p):
        my_vstages = list(range(d, S, p))  # d, d+p, d+2p, ...
        cursors = {vs: 0 for vs in my_vstages}
        remaining = sum(len(vstage_orders[vs]) for vs in my_vstages)
        while remaining > 0:
            for vs in my_vstages:
                c = cursors[vs]
                if c < len(vstage_orders[vs]):
                    unit_idx, phase = vstage_orders[vs][c]
                    per_device[d].append((unit_idx, vs, phase))
                    cursors[vs] = c + 1
                    remaining -= 1
    return per_device


@dataclass
class PipelineResult:
    step_time: float
    bubble_ratio: float
    peak_activation_bytes: float
    per_device_busy: List[float]


def simulate_pipeline(
    unit_stage_times: List[List[Tuple[float, float]]],
    unit_act_bytes: List[float],
    p: int,
    v: int = 1,
    p2p_time: float = 0.0,
) -> PipelineResult:
    """Outer 1F1B(+V) pipeline simulation over pool-wide units.

    unit_stage_times[j][vs] = (fwd_t, bwd_t) for unit j at virtual stage vs.
    Returns step time, measured bubble ratio, peak in-flight activation, busy/dev.
    """
    if v != 1:
        raise NotImplementedError(
            "interleaved virtual stages (v>1) are deferred; only v=1 1F1B is supported"
        )

    m = len(unit_stage_times)
    S = p * v
    if m == 0:
        return PipelineResult(0.0, 0.0, 0.0, [0.0] * p)

    ops: List[SimOp] = []
    # index of the op for (unit j, vstage vs, phase) once created
    op_index: Dict[Tuple[int, int, str], int] = {}

    def add_op(j: int, vs: int, phase: str, duration: float, out_bytes: float) -> int:
        idx = len(ops)
        ops.append(SimOp(
            name=f"u{j}_v{vs}_{phase}", stream=f"dev{vs % p}",
            duration=duration, depends_on=[], output_bytes=out_bytes,
        ))
        op_index[(j, vs, phase)] = idx
        return idx

    # 1) create all ops (forward then backward virtual-stage chains)
    for j in range(m):
        for vs in range(S):
            fwd_t, _ = unit_stage_times[j][vs]
            add_op(j, vs, "F", fwd_t, unit_act_bytes[j])
        for vs in range(S):
            _, bwd_t = unit_stage_times[j][vs]
            add_op(j, vs, "B", bwd_t, 0.0)

    # 2) data-flow dependencies
    for j in range(m):
        for vs in range(1, S):                       # forward chain
            ops[op_index[(j, vs, "F")]].depends_on.append(op_index[(j, vs - 1, "F")])
        # backward starts after the last forward of the unit
        ops[op_index[(j, S - 1, "B")]].depends_on.append(op_index[(j, S - 1, "F")])
        for vs in range(S - 2, -1, -1):              # backward chain
            ops[op_index[(j, vs, "B")]].depends_on.append(op_index[(j, vs + 1, "B")])

    # 3) hard schedule-order edges (encodes 1F1B issue order on each device).
    #
    # Chain each device's events in `pipeline_schedule` order: op[i] depends
    # on op[i-1] for consecutive events on the same device. This makes the
    # per-device execution order structurally deterministic and equal to the
    # intended 1F1B order regardless of op durations (unlike relying on
    # `simulate()`'s FIFO tie-break among topological orders, which only
    # reproduces 1F1B for equal-duration ops). Restricted to v=1 (guarded
    # above), where the standard 1F1B order is realizable and these edges
    # stay acyclic alongside the data-flow edges from step (2).
    sched = pipeline_schedule(m, p, v)
    for d in range(p):
        for pos in range(1, len(sched[d])):
            prev_j, prev_vs, prev_phase = sched[d][pos - 1]
            j, vs, phase = sched[d][pos]
            ops[op_index[(j, vs, phase)]].depends_on.append(
                op_index[(prev_j, prev_vs, prev_phase)]
            )

    sim = simulate(ops)
    # busy time per device = sum of its op durations
    busy = [0.0] * p
    for op in ops:
        busy[int(op.stream[3:])] += op.duration
    step = sim.wall_clock_time
    total_capacity = p * step
    bubble = 1.0 - (sum(busy) / total_capacity) if total_capacity > 0 else 0.0
    return PipelineResult(
        step_time=step,
        bubble_ratio=max(0.0, bubble),
        peak_activation_bytes=sim.peak_activation_bytes,
        per_device_busy=busy,
    )
