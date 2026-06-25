"""Simulator: multi-stream topological simulation with memory tracking.

Given a List[SimOp] from builder.py, runs a DAG simulation where each stream
has its own clock. Tracks peak activation memory using ref-counting.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List

from llm_perf.builder import SimOp


@dataclass
class SimResult:
    """Result of a multi-stream DAG simulation.

    Attributes:
        wall_clock_time: Simulated wall-clock time in seconds.
        weight_bytes: Total deduplicated model weight memory in bytes.
        peak_activation_bytes: Peak live activation memory in bytes.
        total_comm_bytes: Total communication volume in bytes.
    """

    wall_clock_time: float  # seconds
    weight_bytes: float  # deduplicated weight total
    peak_activation_bytes: float  # peak live activation memory
    total_comm_bytes: float  # total communication volume
    compute_time: float = 0
    tp_comm_time: float = 0
    ep_comm_time: float = 0
    dp_comm_time: float = 0
    cp_comm_time: float = 0


def simulate(ops: List[SimOp]) -> SimResult:
    """Multi-stream topological simulation with memory tracking.

    Each stream has its own clock. Ops execute as soon as:
    1. Their stream is free (stream clock)
    2. All dependencies are complete (max of dependency finish times)

    Memory tracking: output_bytes are allocated when op finishes,
    freed when all consumers have completed (ref_count -> 0).

    Args:
        ops: Ordered list of SimOp nodes forming the computation DAG.

    Returns:
        SimResult with wall-clock time, weight bytes, peak activation bytes,
        and total communication bytes.
    """
    if not ops:
        return SimResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    n = len(ops)

    # -----------------------------------------------------------------------
    # 1. Build forward adjacency (dependents[i] = list of ops that depend on i)
    # -----------------------------------------------------------------------
    dependents: Dict[int, List[int]] = defaultdict(list)
    for i, op in enumerate(ops):
        for dep in op.depends_on:
            dependents[dep].append(i)

    # -----------------------------------------------------------------------
    # 2. Topological sort (Kahn's algorithm)
    # -----------------------------------------------------------------------
    in_degree = [len(op.depends_on) for op in ops]
    ready: deque[int] = deque(i for i in range(n) if in_degree[i] == 0)
    topo_order: List[int] = []

    while ready:
        idx = ready.popleft()
        topo_order.append(idx)
        for dep_idx in dependents[idx]:
            in_degree[dep_idx] -= 1
            if in_degree[dep_idx] == 0:
                ready.append(dep_idx)

    assert len(topo_order) == n, (
        f"Cycle detected in op graph: processed {len(topo_order)} of {n} ops"
    )

    # -----------------------------------------------------------------------
    # 3. Multi-clock simulation
    # -----------------------------------------------------------------------
    stream_clock: Dict[str, float] = defaultdict(float)
    # 新增：用于统计各流纯执行耗时之和
    stream_durations: Dict[str, float] = defaultdict(float)
    finish_time = [0.0] * n

    for idx in topo_order:
        op = ops[idx]
        dep_max = max((finish_time[d] for d in op.depends_on), default=0.0)
        start = max(stream_clock[op.stream], dep_max)
        finish_time[idx] = start + op.duration
        stream_clock[op.stream] = finish_time[idx]

        # 【核心修改】：累加该流下所有 op 的纯执行时间
        # 即使 compute 包含 100 个 op，这里也会把它们的 duration 全部加在一起
        stream_durations[op.stream] += op.duration

    wall_clock = max(stream_clock.values()) if stream_clock else 0.0

    # 提取具体的计算和通信时间
    # 注意：这里的 compute_time 指的是算子在 AI Core 上的总有效执行时间
    compute_time = stream_durations.get("compute", 0.0)

    # 汇总各类通信流的时间
    tp_comm_time = stream_durations.get("tp_comm", 0.0)
    ep_comm_time = stream_durations.get("ep_comm", 0.0)
    dp_comm_time = stream_durations.get("dp_comm", 0.0)
    cp_comm_time = stream_durations.get("cp_comm", 0.0)

    # -----------------------------------------------------------------------
    # 4. Weight bytes (sum all — builder only sets weight_bytes on fwd ops)
    # -----------------------------------------------------------------------
    total_weight = sum(op.weight_bytes for op in ops)

    # -----------------------------------------------------------------------
    # 5. Activation memory tracking with ref-counting
    #
    # consumer_count[i]: how many later ops consume op i's output.
    # When a consumer finishes, decrement the producer's ref-count.
    # When ref-count reaches 0, free producer's output_bytes.
    #
    # We process events in finish-time order.  At each event:
    #   a) Allocate output_bytes for the newly finished op.
    #   b) For each dependency of that op, decrement its ref-count and free
    #      if it reaches 0.
    # -----------------------------------------------------------------------
    # Build consumer ref-counts.
    # Prefer explicit op.consumers if set; otherwise derive from dependents.
    consumer_count = [0] * n
    for i, op in enumerate(ops):
        if op.consumers is not None:
            consumer_count[i] = len(op.consumers)
        else:
            consumer_count[i] = len(dependents[i])

    live_activation = 0.0
    peak_activation = 0.0
    freed = [False] * n

    # Process in finish-time order (ties broken by index — deterministic)
    finish_order = sorted(range(n), key=lambda i: (finish_time[i], i))

    for idx in finish_order:
        op = ops[idx]

        # Allocate this op's output
        live_activation += op.output_bytes
        if live_activation > peak_activation:
            peak_activation = live_activation

        # Decrement ref-counts of dependencies; free those that hit 0
        for dep in op.depends_on:
            if not freed[dep]:
                consumer_count[dep] -= 1
                if consumer_count[dep] <= 0:
                    live_activation -= ops[dep].output_bytes
                    freed[dep] = True

    # -----------------------------------------------------------------------
    # 6. Communication bytes
    # -----------------------------------------------------------------------
    total_comm = sum(op.comm_bytes for op in ops)

    return SimResult(
        wall_clock_time=wall_clock,
        weight_bytes=total_weight,
        peak_activation_bytes=peak_activation,
        total_comm_bytes=total_comm,
        compute_time=compute_time,
        tp_comm_time=tp_comm_time,
        ep_comm_time=ep_comm_time,
        dp_comm_time=dp_comm_time,
        cp_comm_time=cp_comm_time
    )
