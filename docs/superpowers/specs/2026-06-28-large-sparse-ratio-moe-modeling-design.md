# Design: Large-sparse-ratio MoE network modeling (node-limited routing + load imbalance)

Date: 2026-06-28
Status: Approved (brainstorming) → ready for implementation plan

## Goal

Give llm-perf the ability to model how a **large sparse ratio** (many experts, e.g.
1k+, with a small activated top_k) drives **interconnect demand** — primarily to
support **network / cluster selection**: how much cross-node all-to-all bandwidth
the design actually requires, and how much hierarchical (node-limited) routing
and load imbalance change that requirement.

The model is a **performance** model: it captures only what changes FLOPs, bytes,
or communication volume. Convergence, expert specialization quality, and routing
algorithm accuracy are explicitly out of scope.

## Scope decisions (from brainstorming)

- **Primary decision served:** network / cluster selection (cross-node all-to-all
  bandwidth, latency, fabric split). Compute-side MFU is secondary.
- **Load imbalance input:** a **scalar `imbalance_factor ≥ 1.0`** (hottest EP rank
  traffic/compute = mean × factor). Not a probability-distribution model.
- **Routing topology:** **node-limited hierarchical routing** (DeepSeek-V3 style):
  each token reaches at most `M` nodes. Dispatch traffic is split into an
  intra-node (NVLink/HCCS) component and an inter-node (NIC) component, with the
  node-limit `M` as a sweepable knob.
- **Deliverable:** model extension + config knobs + a dedicated
  `sweep_sparse_ratio()` report (per-fabric exposed comm vs N / M / imbalance).

## Non-goals (YAGNI for this iteration)

- Fine-grained expert small-GEMM efficiency decay (compute-side MFU drop).
- Probability-distribution / p99 imbalance modeling.
- Attention token sparsity — **already modeled** via `op_dsa_attention`
  (`index_topk`), used by the DeepSeek-V4 config.
- Router softmax-over-1k cost and aux-loss-free bias mechanics.

## Core modeling insight

Current dispatch cross-node traffic = `tokens × top_k × hidden × dtype`
(`op_alltoall_dispatch`). This **over-counts** cross-node bandwidth: when several
of a token's top_k experts live on the **same** destination node, the token
payload crosses the network **once** to that node and is replicated locally over
NVLink/HCCS.

Node-limited routing splits dispatch into two components:

- **Inter-node (NIC):**
  `tokens × E[distinct_dest_nodes] × hidden × dtype`
  where `E[distinct_dest_nodes] = min(top_k, M, nodes_in_ep)`,
  `M = node_limit`, `nodes_in_ep = ceil(ep / devices_per_node)`.
- **Intra-node (NVLink/HCCS):**
  `tokens × (top_k − E[distinct_dest_nodes]) × hidden × dtype`
  (the remaining experts that are co-located on an already-reached node).

Combine is symmetric. This makes the **"hierarchical routing reduces cross-node
traffic"** benefit explicit and sweepable via `M`.

> Alternative considered (rejected): keep `min(top_k, nodes_in_ep)` with no `M`
> knob. Rejected because it cannot answer "what node_limit should we pick", which
> is the primary selection question.

## Load-imbalance amplification

`imbalance_factor ≥ 1.0` = hottest EP rank's traffic/compute relative to the mean.
Because all-to-all is a synchronous collective whose duration is set by the
**slowest participant**:

- Inter-node and intra-node dispatch/combine bytes are each **× imbalance_factor**.
- The `op_moe_ffn` effective `batch_tokens` is **× imbalance_factor** (the hottest
  rank does more expert compute).

`imbalance_factor = 1.0` and `node_limit = 0` together **reproduce current
behavior exactly** (regression-protected).

## Code changes

### `ops.py`
- New `op_alltoall_dispatch_hierarchical(tokens, hidden_size, top_k, ep_size,
  node_limit, nodes_in_ep, imbalance_factor, dtype_bytes)` and a symmetric
  `op_alltoall_combine_hierarchical(...)`. Each returns the **inter-node** and
  **intra-node** byte components (e.g. an `OpCost` per fabric, or a small struct
  with `inter_bytes` / `intra_bytes`). Docstring cites DeepSeek-V3 node-limited
  routing.
- Existing `op_alltoall_dispatch` / `op_alltoall_combine` retained unchanged
  (backward compatibility / regression baseline).

### `builder.py` (MoE branch, ~lines 591–687)
- When `ep > 1`, replace the single dispatch `SimOp` with **two** `SimOp`s:
  one `fabric="nvlink"` (intra-node share, `stream="ep_comm"`) and one
  `fabric="nic"` (inter-node share, `stream="ep_comm"`), each costed via
  `comm_time` with the appropriate `is_intra_node` / `group_size`, so each lands
  on its correct per-fabric clock. Same for combine.
- When the EP group fits in a single node (`nodes_in_ep == 1`), the inter-node
  share is 0 and only the intra-node `SimOp` is emitted.
- Multiply `op_moe_ffn`'s `batch_tokens` by `imbalance_factor`.

### `config.py` (`ParallelismConfig`)
- Add `moe_node_limit: int = 0` (0 = unlimited → degenerate to current behavior).
- Add `moe_imbalance_factor: float = 1.0`.
- Defaults guarantee zero behavior change for existing configs/tests.

## Deliverable: `sweep_sparse_ratio()` + report

- `model.py`: add `sweep_sparse_ratio(grid)` mirroring `compare_precision()`. For
  each point in a grid over `(num_experts, top_k, node_limit M, imbalance_factor,
  ep)`, run the training-step simulation and emit per point: step time,
  **`exposed_comm_by_fabric`** (nic vs nvlink), required cross-node bandwidth
  (GB/s), peak memory, feasibility.
- `report.py`: add side-by-side table formatting, reusing the existing per-fabric
  rendering (`exposed [fabric]: … ms`).

## Testing (TDD)

- `node_limit=0` & `imbalance_factor=1.0` → byte-for-byte equal to existing
  `op_alltoall_dispatch` / `op_alltoall_combine` (regression guard).
- Decreasing `M` → inter-node bytes decrease monotonically, intra-node bytes
  increase, total expert compute unchanged.
- Increasing `imbalance_factor` → exposed comm on both fabrics and hottest-rank
  compute rise proportionally.
- `nodes_in_ep == 1` (EP fits one node) → inter-node share = 0.
- `sweep_sparse_ratio` report has complete fields and correct feasibility flags.
