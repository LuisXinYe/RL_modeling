# Changelog

## [Unreleased]

### Fixed
- `FEASIBLE` status now reflects both time budget AND memory feasibility
- Builder rejects invalid TP/PP configurations instead of silently truncating

### Added
- `--format json` flag for `targets` and `check` CLI commands
- `feasible` field on `TargetReport` (composite of time + memory)
- `feasibility_check` now accepts optional `time_budget_hours`
- CLI error handling: friendly messages instead of raw tracebacks
- ParallelismConfig validates tp/pp/dp/ep/cp >= 1
- Builder validates TP divides num_heads/num_kv_heads/intermediate_size
- Docstrings for all public classes and functions
- `examples/demo.py` — exploration workflow demo
- `docs/architecture.md` — architecture deep-dive
- `docs/config-reference.md` — configuration field reference
- `docs/result-interpretation.md` — report interpretation guide
- `docs/calibration-guide.md` — calibration coefficient guide
- `docs/troubleshooting.md` — common errors and fixes

### Changed
- `format_table` shows `NOT FEASIBLE: OOM`, `NOT FEASIBLE: OVER TIME`, or combined
- `check` command help text now shows hardware shortnames
