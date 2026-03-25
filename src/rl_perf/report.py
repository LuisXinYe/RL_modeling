import json
from dataclasses import asdict, dataclass


@dataclass
class MemoryProfile:
    """Per-device memory breakdown for training and generation phases.

    All sizes are in GB. Feasibility flags indicate whether total fits in HBM.
    """

    weight_gb: float  # Model weight memory in GB (per device, after TP/PP sharding)
    optimizer_gb: float  # Optimizer state memory in GB (Adam master weights + moments)
    activation_peak_gb: float  # Peak activation memory in GB during training
    kv_cache_gb: float  # KV cache memory in GB during generation
    ref_model_gb: float  # Reference model weight memory in GB (0 if offloaded/absent)
    total_train_gb: float  # Total training memory: weights + optimizer + activations + ref
    total_gen_gb: float  # Total generation memory: weights + KV cache
    usable_hbm_gb: float  # Usable HBM capacity in GB (after framework overhead)
    train_feasible: bool  # True if total_train_gb < usable_hbm_gb
    gen_feasible: bool  # True if total_gen_gb < usable_hbm_gb


@dataclass
class TargetReport:
    """Complete performance and feasibility report for one RL epoch."""

    epoch_time_hours: float  # Total epoch wall-clock time in hours
    within_budget: bool  # True if epoch_time_hours <= time_budget_hours
    bottleneck: str  # Which phase is the bottleneck: "generation" or "training"
    bottleneck_slack: float  # Fractional slack of the non-bottleneck phase (0.0-1.0)
    gen_tps_target: float  # Required generation throughput in tokens/s
    train_tps_target: float  # Required training throughput in tokens/s
    gen_samples_per_sec: float  # Generation throughput in samples/s
    train_samples_per_sec: float  # Training throughput in samples/s
    gen_time_hours: float  # Generation phase wall-clock time in hours
    train_time_hours: float  # Training phase wall-clock time in hours
    memory: MemoryProfile  # Per-device memory breakdown
    gen_parallel: object = None  # ParallelismConfig used for generation
    train_parallel: object = None  # ParallelismConfig used for training
    feasible: bool = True  # True if within budget and no OOM


def format_table(report: TargetReport) -> str:
    """Format a TargetReport as a human-readable text table.

    Args:
        report: The TargetReport to format.

    Returns:
        Multi-line string with epoch time, TPS targets, and memory summary.
    """
    reasons = []
    if not report.within_budget:
        reasons.append("OVER TIME")
    if not report.memory.train_feasible or not report.memory.gen_feasible:
        reasons.append("OOM")
    status_str = "FEASIBLE" if not reasons else f"NOT FEASIBLE: {' + '.join(reasons)}"
    mem = report.memory
    lines = [
        "=" * 60,
        "          RL Training Performance Report",
        "=" * 60,
        f" Epoch time:        {report.epoch_time_hours:.2f} hours  [{status_str}]",
        f" Bottleneck:        {report.bottleneck} (slack: {report.bottleneck_slack:.1%})",
        "-" * 60,
        " Generation:",
        f"   TPS target:      {report.gen_tps_target:,.0f} tokens/s",
        f"   Samples/s:       {report.gen_samples_per_sec:.2f}",
        f"   Time:            {report.gen_time_hours:.2f} hours",
        "-" * 60,
        " Training:",
        f"   TPS target:      {report.train_tps_target:,.0f} tokens/s",
        f"   Samples/s:       {report.train_samples_per_sec:.2f}",
        f"   Time:            {report.train_time_hours:.2f} hours",
        "-" * 60,
        " Memory:",
        f"   Train: {mem.total_train_gb:.1f}/{mem.usable_hbm_gb:.1f} GB  [{'OK' if mem.train_feasible else 'OOM'}]",
        f"   Gen:   {mem.total_gen_gb:.1f}/{mem.usable_hbm_gb:.1f} GB  [{'OK' if mem.gen_feasible else 'OOM'}]",
        "=" * 60,
    ]
    return "\n".join(lines)


def format_json(report: TargetReport) -> str:
    """Serialize a TargetReport to a JSON string.

    Args:
        report: The TargetReport to serialize.

    Returns:
        Pretty-printed JSON string (indent=2).
    """
    return json.dumps(asdict(report), indent=2, default=str)
