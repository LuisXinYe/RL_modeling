"""Builder: converts ModelConfig + ParallelismConfig into SimOp sequences.

Parallelism shapes are divided HERE — ops.py knows nothing about TP/EP.
Each op produces a SimOp with name, stream, duration, depends_on,
weight_bytes, and output_bytes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from rl_perf import ops
from rl_perf.config import (
    HardwareConfig,
    LayerConfig,
    ModelConfig,
    ParallelismConfig,
    Phase,
    RLConfig,
    VisionEncoderConfig,
)


def _split_stages(
    all_layers: List[LayerConfig], pp: int
) -> List[List[LayerConfig]]:
    """Split layers into pp pipeline stages, allowing uneven distribution.

    When num_layers is not divisible by pp, the first (num_layers % pp) stages
    get one extra layer — matching Megatron-LM's behavior.

    E.g. 51 layers / pp=8 → [7, 7, 7, 6, 6, 6, 6, 6]
    """
    n = len(all_layers)
    base = n // pp
    remainder = n % pp
    stages = []
    offset = 0
    for i in range(pp):
        count = base + (1 if i < remainder else 0)
        stages.append(all_layers[offset : offset + count])
        offset += count
    return stages


def _validate_parallelism(
    all_layers: List[LayerConfig], parallel_cfg: ParallelismConfig
):
    """Validate parallelism config against model dimensions."""
    tp = parallel_cfg.tp
    pp = parallel_cfg.pp

    if pp > len(all_layers):
        raise ValueError(
            f"pp={pp} exceeds num_layers={len(all_layers)}. "
            f"pp must be <= number of model layers."
        )

    for i, layer in enumerate(all_layers):
        if layer.num_heads % tp != 0:
            raise ValueError(
                f"Layer {i}: num_heads={layer.num_heads} not divisible by tp={tp}. "
                f"Choose tp that evenly divides num_heads."
            )
        if layer.attention in ("GQA", "MHA", "SWA") and layer.num_kv_heads % tp != 0:
            raise ValueError(
                f"Layer {i}: num_kv_heads={layer.num_kv_heads} not divisible by tp={tp}. "
                f"Choose tp that evenly divides num_kv_heads."
            )
        if layer.ffn == "SwiGLU" and layer.intermediate_size % tp != 0:
            raise ValueError(
                f"Layer {i}: intermediate_size={layer.intermediate_size} not divisible by tp={tp}. "
                f"Choose tp that evenly divides intermediate_size."
            )


@dataclass
class SimOp:
    name: str
    stream: str  # "compute", "tp_comm", "ep_comm", "dp_comm"
    duration: float  # seconds
    depends_on: List[int] = field(default_factory=list)
    weight_bytes: float = 0
    output_bytes: float = 0
    comm_bytes: float = 0  # communication volume in bytes (for comm ops)
    consumers: Optional[List[int]] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_intra_node(group_size: int, hw: HardwareConfig) -> bool:
    return group_size <= hw.devices_per_node


_COLLECTIVE_OPS = {
    "allreduce": (ops.op_allreduce, "ring"),
    "allgather": (ops.op_allgather, "ring_half"),
    "reducescatter": (ops.op_reducescatter, "ring_half"),
}


def _build_tp_comm(
    name: str,
    collective: str,
    batch: int,
    seq_len: int,
    hidden_size: int,
    dtype_bytes: int,
    tp: int,
    hw: HardwareConfig,
    dep_idx: int,
) -> SimOp:
    """Build a TP collective comm SimOp (allreduce, allgather, or reducescatter).

    Note: seq_len should already be CP-local (i.e. seq_len // cp) when called
    from build_layer_ops, which handles CP sharding internally.
    """
    op_fn, algorithm = _COLLECTIVE_OPS[collective]
    msg = batch * seq_len * hidden_size * dtype_bytes
    cost = op_fn(msg, group_size=tp)
    duration = ops.comm_time(
        cost,
        hw,
        group_size=tp,
        is_intra_node=_is_intra_node(tp, hw),
        algorithm=algorithm,
    )
    return SimOp(
        name=name,
        stream="tp_comm",
        duration=duration,
        depends_on=[dep_idx],
        weight_bytes=0,
        output_bytes=0,
        comm_bytes=cost.comm_bytes,
    )


# ---------------------------------------------------------------------------
# build_layer_ops
# ---------------------------------------------------------------------------


def build_layer_ops(
    layer_cfg: LayerConfig,
    model_cfg: ModelConfig,
    parallel_cfg: ParallelismConfig,
    hw: HardwareConfig,
    batch: int,
    seq_len: int,
    phase: Phase,
    kv_len: Optional[int] = None,
    index_offset: int = 0,
) -> List[SimOp]:
    """Build SimOps for one transformer layer.

    Parameters
    ----------
    index_offset:
        Global index of the first op returned (used so depends_on indices are
        globally consistent when ops from multiple layers are concatenated).
    """
    tp = parallel_cfg.tp
    ep = parallel_cfg.ep
    cp = parallel_cfg.cp
    sp = parallel_cfg.sp
    d = model_cfg.hidden_size
    dtype_bytes = model_cfg.dtype_bytes
    # CP shards the sequence across ranks; all per-rank ops use the local length
    local_seq_len = seq_len // cp if cp > 1 else seq_len
    # SP further shards the sequence across TP ranks (only when sp=True and tp>1)
    sp_seq_len = local_seq_len // tp if (sp and tp > 1) else local_seq_len

    # batch_tokens for ops that run on SP-sharded data (rmsnorm, residual, etc.)
    sp_batch_tokens = batch * sp_seq_len
    # batch_tokens for ops that run on full CP-local data (attention, FFN)
    batch_tokens = batch * local_seq_len

    result: List[SimOp] = []

    def _idx(local: int) -> int:
        """Convert local (within-layer) index to global index."""
        return index_offset + local

    # ---- 0. Pre-attention RMSNorm ----------------------------------------
    # RMSNorm runs on SP-sharded data (seq_len/cp/tp when SP on, seq_len/cp otherwise)
    norm1_cost = ops.op_rmsnorm(d, sp_batch_tokens, phase, dtype_bytes)
    norm1 = SimOp(
        name="rmsnorm_pre_attn",
        stream="compute",
        duration=ops.roofline_time(norm1_cost, hw, is_large_gemm=False),
        depends_on=[],  # first op in layer; caller chains layers together
        weight_bytes=norm1_cost.weight_bytes,
        output_bytes=norm1_cost.output_bytes,
    )
    result.append(norm1)  # local idx 0

    # ---- SP AllGather (before CP ring / attention) ----------------------------
    # SP AllGather restores sequence from sp_seq_len to local_seq_len (CP-local).
    # Must happen BEFORE CP ring, because CP ring operates on full CP-local sequence.
    if sp and tp > 1:
        ag_attn = _build_tp_comm(
            name="tp_allgather_attn",
            collective="allgather",
            batch=batch,
            seq_len=sp_seq_len,
            hidden_size=d,
            dtype_bytes=dtype_bytes,
            tp=tp,
            hw=hw,
            dep_idx=_idx(len(result) - 1),  # depends on norm1
        )
        result.append(ag_attn)

    # ---- CP Ring comm (before attention) -------------------------------------
    # CP Ring operates on the full CP-local sequence (after SP AllGather if SP is on).
    if cp > 1:
        attn_type = layer_cfg.attention
        if attn_type == "MLA":
            kv_dim = layer_cfg.kv_compression_dim + layer_cfg.rope_dim
        elif attn_type in ("GQA", "MHA", "SWA"):
            tp_kv_heads = layer_cfg.num_kv_heads // tp
            kv_dim = tp_kv_heads * layer_cfg.head_dim
        else:
            kv_dim = layer_cfg.num_kv_heads * layer_cfg.head_dim

        cp_cost = ops.op_ring_cp(seq_len, cp, kv_dim, dtype_bytes)
        cp_ring = SimOp(
            name="cp_ring_kv",
            stream="cp_comm",
            duration=ops.comm_time(
                cp_cost,
                hw,
                group_size=cp,
                is_intra_node=_is_intra_node(cp, hw),
                algorithm="ring_half",
            ),
            depends_on=[_idx(len(result) - 1)],  # depends on SP AG or norm1
            weight_bytes=0,
            output_bytes=0,
            comm_bytes=cp_cost.comm_bytes,
        )
        result.append(cp_ring)

    # ---- 1. Attention -------------------------------------------------------
    attn_type = layer_cfg.attention

    if attn_type in ("GQA", "MHA"):
        tp_num_heads = layer_cfg.num_heads // tp
        tp_kv_heads = layer_cfg.num_kv_heads // tp
        attn_cost = ops.op_gqa_attention(
            num_heads=tp_num_heads,
            num_kv_heads=tp_kv_heads,
            head_dim=layer_cfg.head_dim,
            hidden_size=d,
            batch=batch,
            seq_len=local_seq_len,
            phase=phase,
            kv_len=kv_len,
            dtype_bytes=dtype_bytes,
        )
    elif attn_type == "SWA":
        tp_num_heads = layer_cfg.num_heads // tp
        tp_kv_heads = layer_cfg.num_kv_heads // tp
        attn_cost = ops.op_swa_attention(
            num_heads=tp_num_heads,
            num_kv_heads=tp_kv_heads,
            head_dim=layer_cfg.head_dim,
            hidden_size=d,
            batch=batch,
            seq_len=local_seq_len,
            phase=phase,
            window_size=layer_cfg.window_size,
            kv_len=kv_len,
            dtype_bytes=dtype_bytes,
        )
    elif attn_type == "MLA":
        tp_num_heads = layer_cfg.num_heads // tp
        attn_cost = ops.op_mla_attention(
            hidden_size=d,
            num_heads=tp_num_heads,
            head_dim=layer_cfg.head_dim,
            kv_compression_dim=layer_cfg.kv_compression_dim,
            query_compression_dim=layer_cfg.query_compression_dim,
            rope_dim=layer_cfg.rope_dim,
            batch=batch,
            seq_len=local_seq_len,
            phase=phase,
            kv_len=kv_len,
            dtype_bytes=dtype_bytes,
        )
    else:
        raise ValueError(f"Unknown attention type: {attn_type}")

    attn_dep_idx = _idx(len(result) - 1)  # depends on CP ring / SP AG / norm1
    attn_op = SimOp(
        name=f"attention_{attn_type.lower()}",
        stream="compute",
        duration=ops.roofline_time(attn_cost, hw, is_large_gemm=True),
        depends_on=[attn_dep_idx],
        weight_bytes=attn_cost.weight_bytes,
        output_bytes=attn_cost.output_bytes,
    )
    result.append(attn_op)

    # ---- 2. TP comm (attention) ---------------------------------------------
    if tp > 1:
        # SP: ReduceScatter (returns to SP-sharded state)
        # No SP: AllReduce
        collective = "reducescatter" if sp else "allreduce"
        tp_attn_comm = _build_tp_comm(
            name=f"tp_{collective}_attn",
            collective=collective,
            batch=batch,
            seq_len=local_seq_len,
            hidden_size=d,
            dtype_bytes=dtype_bytes,
            tp=tp,
            hw=hw,
            dep_idx=_idx(len(result) - 1),
        )
        result.append(tp_attn_comm)
    last_compute_idx = _idx(len(result) - 1)

    # ---- 3. Pre-FFN RMSNorm -------------------------------------------------
    # After TP ReduceScatter (SP), data is back to SP-sharded state.
    # So RMSNorm runs on SP-sharded data.
    norm2_cost = ops.op_rmsnorm(d, sp_batch_tokens, phase, dtype_bytes)
    norm2 = SimOp(
        name="rmsnorm_pre_ffn",
        stream="compute",
        duration=ops.roofline_time(norm2_cost, hw, is_large_gemm=False),
        depends_on=[last_compute_idx],
        weight_bytes=norm2_cost.weight_bytes,
        output_bytes=norm2_cost.output_bytes,
    )
    result.append(norm2)  # local idx = len(result)-1
    last_compute_local = len(result) - 1

    # ---- 4. FFN -------------------------------------------------------------
    # SP: insert AllGather before FFN to restore from sp_seq_len to local_seq_len
    if sp and tp > 1:
        ag_ffn = _build_tp_comm(
            name="tp_allgather_ffn",
            collective="allgather",
            batch=batch,
            seq_len=sp_seq_len,
            hidden_size=d,
            dtype_bytes=dtype_bytes,
            tp=tp,
            hw=hw,
            dep_idx=_idx(len(result) - 1),  # depends on norm2
        )
        result.append(ag_ffn)
        last_compute_local = len(result) - 1

    ffn_type = layer_cfg.ffn

    if ffn_type == "SwiGLU":
        tp_intermediate = layer_cfg.intermediate_size // tp
        ffn_cost = ops.op_swiglu_ffn(
            hidden_size=d,
            intermediate_size=tp_intermediate,
            batch_tokens=batch_tokens,
            phase=phase,
            dtype_bytes=dtype_bytes,
        )
        ffn_op = SimOp(
            name="ffn_swiglu",
            stream="compute",
            duration=ops.roofline_time(ffn_cost, hw, is_large_gemm=True),
            depends_on=[_idx(last_compute_local)],
            weight_bytes=ffn_cost.weight_bytes,
            output_bytes=ffn_cost.output_bytes,
        )
        result.append(ffn_op)
        last_compute_local = len(result) - 1

    elif ffn_type == "MoE":
        ep_num_experts = max(1, layer_cfg.num_experts // ep)

        # EP AllToAll dispatch: send tokens to target expert EP ranks
        if ep > 1:
            a2a_dispatch_cost = ops.op_alltoall_dispatch(
                tokens=batch_tokens,
                hidden_size=d,
                top_k=layer_cfg.top_k,
                ep_size=ep,
                dtype_bytes=dtype_bytes,
            )
            ep_dispatch = SimOp(
                name="ep_alltoall_dispatch",
                stream="ep_comm",
                duration=ops.comm_time(
                    a2a_dispatch_cost,
                    hw,
                    group_size=ep,
                    is_intra_node=_is_intra_node(ep, hw),
                    algorithm="alltoall",
                ),
                depends_on=[_idx(last_compute_local)],
                weight_bytes=0,
                output_bytes=0,
                comm_bytes=a2a_dispatch_cost.comm_bytes,
            )
            result.append(ep_dispatch)
            last_compute_local = len(result) - 1

        ffn_cost = ops.op_moe_ffn(
            hidden_size=d,
            expert_intermediate_size=layer_cfg.expert_intermediate_size,
            num_experts=ep_num_experts,
            num_shared_experts=layer_cfg.num_shared_experts,
            shared_intermediate_size=layer_cfg.shared_intermediate_size,
            top_k=layer_cfg.top_k,
            batch_tokens=batch_tokens,
            phase=phase,
            dtype_bytes=dtype_bytes,
        )
        ffn_op = SimOp(
            name="ffn_moe",
            stream="compute",
            duration=ops.roofline_time(ffn_cost, hw, is_large_gemm=True),
            depends_on=[_idx(last_compute_local)],
            weight_bytes=ffn_cost.weight_bytes,
            output_bytes=ffn_cost.output_bytes,
        )
        result.append(ffn_op)
        last_compute_local = len(result) - 1

        # EP AllToAll combine: gather expert outputs back to original EP ranks
        if ep > 1:
            a2a_combine_cost = ops.op_alltoall_combine(
                tokens=batch_tokens,
                hidden_size=d,
                top_k=layer_cfg.top_k,
                ep_size=ep,
                dtype_bytes=dtype_bytes,
            )
            ep_combine = SimOp(
                name="ep_alltoall_combine",
                stream="ep_comm",
                duration=ops.comm_time(
                    a2a_combine_cost,
                    hw,
                    group_size=ep,
                    is_intra_node=_is_intra_node(ep, hw),
                    algorithm="alltoall",
                ),
                depends_on=[_idx(last_compute_local)],
                weight_bytes=0,
                output_bytes=0,
                comm_bytes=a2a_combine_cost.comm_bytes,
            )
            result.append(ep_combine)
            last_compute_local = len(result) - 1
    else:
        raise ValueError(f"Unknown FFN type: {ffn_type}")

    # ---- 5. TP comm (FFN) ----------------------------------------------------
    if tp > 1:
        collective = "reducescatter" if sp else "allreduce"
        tp_ffn_comm = _build_tp_comm(
            name=f"tp_{collective}_ffn",
            collective=collective,
            batch=batch,
            seq_len=local_seq_len,
            hidden_size=d,
            dtype_bytes=dtype_bytes,
            tp=tp,
            hw=hw,
            dep_idx=_idx(last_compute_local),
        )
        result.append(tp_ffn_comm)
        last_compute_local = len(result) - 1

    # ---- 6. mHC residual (optional) -----------------------------------------
    # After FFN TP ReduceScatter (SP), data is back to SP-sharded state.
    if layer_cfg.residual == "mHC":
        mhc_cost = ops.op_mhc_residual(
            hidden_size=d,
            expansion=layer_cfg.mhc_expansion,
            batch_tokens=sp_batch_tokens,
            phase=phase,
            dtype_bytes=dtype_bytes,
        )
        mhc_op = SimOp(
            name="mhc_residual",
            stream="compute",
            duration=ops.roofline_time(mhc_cost, hw, is_large_gemm=False),
            depends_on=[_idx(last_compute_local)],
            weight_bytes=mhc_cost.weight_bytes,
            output_bytes=mhc_cost.output_bytes,
        )
        result.append(mhc_op)

    # Zero out weight_bytes for BWD ops to avoid double-counting in simulator
    if phase == Phase.TRAIN_BWD:
        for op in result:
            op.weight_bytes = 0

    return result


# ---------------------------------------------------------------------------
# Parameter estimation
# ---------------------------------------------------------------------------


def _estimate_param_count(
    model_cfg: ModelConfig,
    parallel_cfg: ParallelismConfig,
    all_layers: List[LayerConfig],
) -> int:
    """Quick estimate of model parameters per TP shard for gradient sync sizing."""
    tp = parallel_cfg.tp
    d = model_cfg.hidden_size
    dtype_bytes = model_cfg.dtype_bytes

    param_bytes = 0
    for layer_cfg in all_layers:
        # Attention
        attn_type = layer_cfg.attention
        if attn_type in ("GQA", "MHA", "SWA"):
            tp_num_heads = layer_cfg.num_heads // tp
            tp_kv_heads = layer_cfg.num_kv_heads // tp
            # Q: d*num_heads*head_dim; K,V: d*kv_heads*head_dim
            # O: RowParallel, input dim = tp_num_heads * head_dim
            q_params = d * tp_num_heads * layer_cfg.head_dim
            kv_params = 2 * d * tp_kv_heads * layer_cfg.head_dim
            o_params = tp_num_heads * layer_cfg.head_dim * d
            param_bytes += (q_params + kv_params + o_params) * dtype_bytes
        elif attn_type == "MLA":
            w_b = (
                d * layer_cfg.query_compression_dim
                + layer_cfg.query_compression_dim * d
                + d * layer_cfg.kv_compression_dim
                + layer_cfg.kv_compression_dim * d
                + layer_cfg.kv_compression_dim * d
                + d * d
            ) * dtype_bytes
            param_bytes += w_b

        # FFN
        ffn_type = layer_cfg.ffn
        if ffn_type == "SwiGLU":
            tp_intermediate = layer_cfg.intermediate_size // tp
            param_bytes += 3 * d * tp_intermediate * dtype_bytes
        elif ffn_type == "MoE":
            ep = parallel_cfg.ep
            ep_num_experts = max(1, layer_cfg.num_experts // ep)
            param_bytes += (
                ep_num_experts
                * 3
                * d
                * layer_cfg.expert_intermediate_size
                * dtype_bytes
            )
            # Shared experts (not EP-partitioned)
            if layer_cfg.num_shared_experts > 0:
                shared_int = (
                    layer_cfg.shared_intermediate_size or layer_cfg.intermediate_size
                )
                param_bytes += (
                    layer_cfg.num_shared_experts * 3 * d * shared_int * dtype_bytes
                )

        # RMSNorm (small, 2 per layer)
        param_bytes += 2 * d * dtype_bytes

    return int(param_bytes / dtype_bytes)  # return as element count


def _estimate_vision_encoder_params(ve_cfg: VisionEncoderConfig) -> int:
    """Estimate parameter count for a Vision Transformer encoder.

    Per layer: Q proj + KV proj + output proj + SwiGLU FFN (gate + up + down).
    """
    d = ve_cfg.hidden_size
    d_ff = ve_cfg.intermediate_size
    n_layers = ve_cfg.num_layers
    n_heads = ve_cfg.num_heads
    n_kv = ve_cfg.get_num_kv_heads()
    head_dim = ve_cfg.get_head_dim()

    q_params = d * (n_heads * head_dim)
    kv_params = 2 * d * (n_kv * head_dim)
    out_params = (n_heads * head_dim) * d
    ffn_params = 3 * d * d_ff  # SwiGLU: gate + up + down

    return n_layers * (q_params + kv_params + out_params + ffn_params)


# ---------------------------------------------------------------------------
# build_vision_encoder_step
# ---------------------------------------------------------------------------


def build_vision_encoder_step(
    model_cfg: ModelConfig,
    hw: HardwareConfig,
    parallel_cfg: ParallelismConfig,
    batch_size: int,
) -> List[SimOp]:
    """Build SimOps for the Vision Encoder (ViT) forward + backward pass.

    The ViT runs before the LLM forward pass: images -> patches -> ViT ->
    projection -> LLM. It uses MHA (num_kv_heads = num_heads), standard
    SwiGLU FFN, no MoE.

    The ViT has full recompute (visual_recompute_num_layers = all), so:
    - Forward pass produces image embeddings
    - Backward recomputes forward then computes gradients (1.3x penalty applied
      in pipeline.py via _apply_step_overheads with full_recomputation)

    The sequence length for ViT is the image_seq_len (number of image patches).
    """
    ve_cfg = model_cfg.vision_encoder
    assert ve_cfg is not None

    tp = parallel_cfg.tp
    dtype_bytes = model_cfg.dtype_bytes

    # Build a synthetic LayerConfig from VisionEncoderConfig
    ve_layer = LayerConfig(
        attention="MHA",
        num_heads=ve_cfg.num_heads,
        num_kv_heads=ve_cfg.get_num_kv_heads(),
        head_dim=ve_cfg.get_head_dim(),
        ffn="SwiGLU",
        intermediate_size=ve_cfg.intermediate_size,
    )

    # Build a synthetic ModelConfig with ViT hidden_size for build_layer_ops
    ve_model_cfg = ModelConfig(
        name=f"{model_cfg.name}_vit",
        hidden_size=ve_cfg.hidden_size,
        vocab_size=1,  # unused for ViT
        num_layers=ve_cfg.num_layers,
        dtype=ve_cfg.dtype,
        default_layer=ve_layer,
    )

    batch = batch_size
    seq_len = ve_cfg.image_seq_len()

    all_ops: List[SimOp] = []
    prev_dep: Optional[int] = None

    # ------ Forward pass -------------------------------------------------------
    for i in range(ve_cfg.num_layers):
        offset = len(all_ops)
        layer_ops = build_layer_ops(
            layer_cfg=ve_layer,
            model_cfg=ve_model_cfg,
            parallel_cfg=parallel_cfg,
            hw=hw,
            batch=batch,
            seq_len=seq_len,
            phase=Phase.TRAIN_FWD,
            index_offset=offset,
        )
        if prev_dep is not None and layer_ops:
            layer_ops[0].depends_on = [prev_dep]
        all_ops.extend(layer_ops)
        if layer_ops:
            prev_dep = len(all_ops) - 1

    # ------ Backward pass (reversed layers, with full recompute) ---------------
    for i in range(ve_cfg.num_layers):
        offset = len(all_ops)
        layer_ops = build_layer_ops(
            layer_cfg=ve_layer,
            model_cfg=ve_model_cfg,
            parallel_cfg=parallel_cfg,
            hw=hw,
            batch=batch,
            seq_len=seq_len,
            phase=Phase.TRAIN_BWD,
            index_offset=offset,
        )
        if prev_dep is not None and layer_ops:
            layer_ops[0].depends_on = [prev_dep]
        all_ops.extend(layer_ops)
        if layer_ops:
            prev_dep = len(all_ops) - 1

    # Prefix all ops with "vit_"
    for op in all_ops:
        op.name = f"vit_{op.name}"

    return all_ops


def build_vision_encoder_step_fwd(
    model_cfg: ModelConfig,
    hw: HardwareConfig,
    parallel_cfg: ParallelismConfig,
    batch_size: int,
) -> List[SimOp]:
    """Build SimOps for the Vision Encoder (ViT) forward pass.

    The ViT runs before the LLM forward pass: images -> patches -> ViT ->
    projection -> LLM. It uses MHA (num_kv_heads = num_heads), standard
    SwiGLU FFN, no MoE.

    The ViT has full recompute (visual_recompute_num_layers = all), so:
    - Forward pass produces image embeddings
    - Backward recomputes forward then computes gradients (1.3x penalty applied
      in pipeline.py via _apply_step_overheads with full_recomputation)

    The sequence length for ViT is the image_seq_len (number of image patches).
    """
    ve_cfg = model_cfg.vision_encoder
    assert ve_cfg is not None

    tp = parallel_cfg.tp
    dtype_bytes = model_cfg.dtype_bytes

    # Build a synthetic LayerConfig from VisionEncoderConfig
    ve_layer = LayerConfig(
        attention="MHA",
        num_heads=ve_cfg.num_heads,
        num_kv_heads=ve_cfg.get_num_kv_heads(),
        head_dim=ve_cfg.get_head_dim(),
        ffn="SwiGLU",
        intermediate_size=ve_cfg.intermediate_size,
    )

    # Build a synthetic ModelConfig with ViT hidden_size for build_layer_ops
    ve_model_cfg = ModelConfig(
        name=f"{model_cfg.name}_vit",
        hidden_size=ve_cfg.hidden_size,
        vocab_size=1,  # unused for ViT
        num_layers=ve_cfg.num_layers,
        dtype=ve_cfg.dtype,
        default_layer=ve_layer,
    )

    batch = batch_size
    seq_len = ve_cfg.image_seq_len()

    all_ops: List[SimOp] = []
    prev_dep: Optional[int] = None

    # ------ Forward pass -------------------------------------------------------
    for i in range(ve_cfg.num_layers):
        offset = len(all_ops)
        layer_ops = build_layer_ops(
            layer_cfg=ve_layer,
            model_cfg=ve_model_cfg,
            parallel_cfg=parallel_cfg,
            hw=hw,
            batch=batch,
            seq_len=seq_len,
            phase=Phase.TRAIN_FWD,
            index_offset=offset,
        )
        if prev_dep is not None and layer_ops:
            layer_ops[0].depends_on = [prev_dep]
        all_ops.extend(layer_ops)
        if layer_ops:
            prev_dep = len(all_ops) - 1

    # Prefix all ops with "vit_"
    for op in all_ops:
        op.name = f"vit_{op.name}"

    return all_ops

# ---------------------------------------------------------------------------
# build_forward_pass
# ---------------------------------------------------------------------------


def build_forward_pass(
    model_cfg: ModelConfig,
    hw: HardwareConfig,
    parallel_cfg: ParallelismConfig,
    rl_cfg: RLConfig,
    include_mtp: bool = True,
    name_prefix: str = "",
    stage_layers: Optional[List[LayerConfig]] = None,
) -> List[SimOp]:
    """Build SimOps for a forward-pass-only (no backward, no DP sync, no optimizer).

    Used for reference model forward.

    Parameters
    ----------
    include_mtp: Whether to include MTP head forward ops.
    name_prefix: Prefix for op names to disambiguate (e.g. "ref_").
    stage_layers: Optional per-stage layer list. If None, uses _split_stages()[0].
    """
    all_layers = model_cfg.get_layers()
    _validate_parallelism(all_layers, parallel_cfg)
    pp = parallel_cfg.pp
    dtype_bytes = model_cfg.dtype_bytes

    seq_len = rl_cfg.avg_prompt_len + rl_cfg.avg_response_len
    batch = rl_cfg.train_micro_batch_size
    cp = parallel_cfg.cp
    local_seq_len = seq_len // cp if cp > 1 else seq_len

    if stage_layers is None:
        stage_layers = _split_stages(all_layers, pp)[0]

    all_ops: List[SimOp] = []
    prev_dep: Optional[int] = None

    # ------ Forward pass -------------------------------------------------------
    for layer_cfg in stage_layers:
        offset = len(all_ops)
        layer_ops = build_layer_ops(
            layer_cfg=layer_cfg,
            model_cfg=model_cfg,
            parallel_cfg=parallel_cfg,
            hw=hw,
            batch=batch,
            seq_len=seq_len,
            phase=Phase.TRAIN_FWD,
            index_offset=offset,
        )
        if prev_dep is not None and layer_ops:
            layer_ops[0].depends_on = [prev_dep]
        all_ops.extend(layer_ops)
        if layer_ops:
            prev_dep = len(all_ops) - 1

    # ------ MTP head (forward only) -------------------------------------------
    if include_mtp:
        mtp_depth = 0
        if model_cfg.auxiliary:
            mtp_depth = model_cfg.auxiliary.get("mtp_depth", 0)

        if mtp_depth > 0:
            mtp_fwd_cost = ops.op_mtp_head(
                hidden_size=model_cfg.hidden_size,
                vocab_size=model_cfg.vocab_size,
                mtp_depth=mtp_depth,
                batch_tokens=batch * local_seq_len,
                phase=Phase.TRAIN_FWD,
                dtype_bytes=dtype_bytes,
            )
            mtp_fwd = SimOp(
                name=f"{name_prefix}mtp_head_fwd",
                stream="compute",
                duration=ops.roofline_time(mtp_fwd_cost, hw, is_large_gemm=True),
                depends_on=[prev_dep] if prev_dep is not None else [],
                weight_bytes=mtp_fwd_cost.weight_bytes,
                output_bytes=mtp_fwd_cost.output_bytes,
            )
            all_ops.append(mtp_fwd)
            prev_dep = len(all_ops) - 1

    # Apply name prefix
    if name_prefix:
        for op in all_ops:
            op.name = f"{name_prefix}{op.name}"

    return all_ops


# ---------------------------------------------------------------------------
# build_training_step
# ---------------------------------------------------------------------------


def build_training_step(
    model_cfg: ModelConfig,
    hw: HardwareConfig,
    parallel_cfg: ParallelismConfig,
    rl_cfg: RLConfig,
    stage_layers: Optional[List[LayerConfig]] = None,
) -> List[SimOp]:
    """Build SimOps for one full training micro-step (forward + backward + DP sync + optim).

    Parameters
    ----------
    stage_layers: Optional per-stage layer list. If None, uses _split_stages()[0].
    """
    all_layers = model_cfg.get_layers()
    _validate_parallelism(all_layers, parallel_cfg)
    pp = parallel_cfg.pp
    dp = parallel_cfg.dp
    dtype_bytes = model_cfg.dtype_bytes

    seq_len = rl_cfg.avg_prompt_len + rl_cfg.avg_response_len
    batch = rl_cfg.train_micro_batch_size
    cp = parallel_cfg.cp
    local_seq_len = seq_len // cp if cp > 1 else seq_len

    # PP stage layers: use provided or default to stage 0
    if stage_layers is None:
        stage_layers = _split_stages(all_layers, pp)[0]

    all_ops: List[SimOp] = []

    # Dependency chain: track the global index of the last op in the main sequence
    prev_dep: Optional[int] = None

    # ------ Forward pass -------------------------------------------------------
    for layer_cfg in stage_layers:
        offset = len(all_ops)
        layer_ops = build_layer_ops(
            layer_cfg=layer_cfg,
            model_cfg=model_cfg,
            parallel_cfg=parallel_cfg,
            hw=hw,
            batch=batch,
            seq_len=seq_len,
            phase=Phase.TRAIN_FWD,
            index_offset=offset,
        )
        # Chain: first op of this layer depends on last op of previous layer
        if prev_dep is not None and layer_ops:
            layer_ops[0].depends_on = [prev_dep]
        all_ops.extend(layer_ops)
        if layer_ops:
            prev_dep = len(all_ops) - 1

    # ------ MTP head (forward) ------------------------------------------------
    mtp_depth = 0
    if model_cfg.auxiliary:
        mtp_depth = model_cfg.auxiliary.get("mtp_depth", 0)

    if mtp_depth > 0:
        mtp_fwd_cost = ops.op_mtp_head(
            hidden_size=model_cfg.hidden_size,
            vocab_size=model_cfg.vocab_size,
            mtp_depth=mtp_depth,
            batch_tokens=batch * local_seq_len,
            phase=Phase.TRAIN_FWD,
            dtype_bytes=dtype_bytes,
        )
        mtp_fwd = SimOp(
            name="mtp_head_fwd",
            stream="compute",
            duration=ops.roofline_time(mtp_fwd_cost, hw, is_large_gemm=True),
            depends_on=[prev_dep] if prev_dep is not None else [],
            weight_bytes=mtp_fwd_cost.weight_bytes,
            output_bytes=mtp_fwd_cost.output_bytes,
        )
        all_ops.append(mtp_fwd)
        prev_dep = len(all_ops) - 1

    # ------ MTP head (backward) — before layer backward (backprop output→input)
    if mtp_depth > 0:
        mtp_bwd_cost = ops.op_mtp_head(
            hidden_size=model_cfg.hidden_size,
            vocab_size=model_cfg.vocab_size,
            mtp_depth=mtp_depth,
            batch_tokens=batch * local_seq_len,
            phase=Phase.TRAIN_BWD,
            dtype_bytes=dtype_bytes,
        )
        mtp_bwd = SimOp(
            name="mtp_head_bwd",
            stream="compute",
            duration=ops.roofline_time(mtp_bwd_cost, hw, is_large_gemm=True),
            depends_on=[prev_dep] if prev_dep is not None else [],
            weight_bytes=0,
            output_bytes=0,
        )
        all_ops.append(mtp_bwd)
        prev_dep = len(all_ops) - 1

    # ------ Backward pass (reversed layers) ------------------------------------
    for layer_cfg in reversed(stage_layers):
        offset = len(all_ops)
        layer_ops = build_layer_ops(
            layer_cfg=layer_cfg,
            model_cfg=model_cfg,
            parallel_cfg=parallel_cfg,
            hw=hw,
            batch=batch,
            seq_len=seq_len,
            phase=Phase.TRAIN_BWD,
            index_offset=offset,
        )
        if prev_dep is not None and layer_ops:
            layer_ops[0].depends_on = [prev_dep]
        all_ops.extend(layer_ops)
        if layer_ops:
            prev_dep = len(all_ops) - 1

    # ------ DP gradient sync ---------------------------------------------------
    param_count = _estimate_param_count(model_cfg, parallel_cfg, stage_layers)
    if dp > 1:
        grad_bytes = param_count * dtype_bytes  # gradient same dtype as weights

        if parallel_cfg.zero_stage < 3:
            # AllReduce (ring)
            dp_cost = ops.op_allreduce(grad_bytes, group_size=dp)
        else:
            # ZeRO-3: ReduceScatter
            dp_cost = ops.op_reducescatter(grad_bytes, group_size=dp)

        dp_algorithm = "ring_half" if parallel_cfg.zero_stage >= 3 else "ring"
        dp_duration = ops.comm_time(
            dp_cost,
            hw,
            group_size=dp,
            is_intra_node=_is_intra_node(dp, hw),
            algorithm=dp_algorithm,
        )
        dp_sync = SimOp(
            name="dp_grad_sync",
            stream="dp_comm",
            duration=dp_duration,
            depends_on=[prev_dep] if prev_dep is not None else [],
            weight_bytes=0,
            output_bytes=0,
            comm_bytes=dp_cost.comm_bytes,
        )
        all_ops.append(dp_sync)
        prev_dep = len(all_ops) - 1

    # ------ Optimizer step (placeholder) ---------------------------------------
    optim_duration = param_count * 1e-10  # rough placeholder
    optim_op = SimOp(
        name="optimizer_step",
        stream="compute",
        duration=optim_duration,
        depends_on=[prev_dep] if prev_dep is not None else [],
        weight_bytes=0,
        output_bytes=0,
    )
    all_ops.append(optim_op)

    return all_ops


# ---------------------------------------------------------------------------
# build_generation_step
# ---------------------------------------------------------------------------


def build_generation_step(
    model_cfg: ModelConfig,
    hw: HardwareConfig,
    parallel_cfg: ParallelismConfig,
    rl_cfg: RLConfig,
    stage_layers: Optional[List[LayerConfig]] = None,
) -> Tuple[List[SimOp], List[SimOp]]:
    """Build SimOps for generation (prefill + decode-per-token).

    Returns (prefill_ops, decode_per_token_ops).

    Parameters
    ----------
    stage_layers: Optional per-stage layer list. If None, uses _split_stages()[0].
    """
    all_layers = model_cfg.get_layers()
    _validate_parallelism(all_layers, parallel_cfg)
    pp = parallel_cfg.pp

    if stage_layers is None:
        stage_layers = _split_stages(all_layers, pp)[0]

    batch = rl_cfg.gen_batch_size
    prompt_len = rl_cfg.avg_prompt_len
    kv_len = rl_cfg.avg_prompt_len + rl_cfg.avg_response_len

    # ------ Prefill ------------------------------------------------------------
    prefill_ops: List[SimOp] = []
    prev_dep: Optional[int] = None

    for layer_cfg in stage_layers:
        offset = len(prefill_ops)
        layer_ops = build_layer_ops(
            layer_cfg=layer_cfg,
            model_cfg=model_cfg,
            parallel_cfg=parallel_cfg,
            hw=hw,
            batch=batch,
            seq_len=prompt_len,
            phase=Phase.PREFILL,
            kv_len=None,
            index_offset=offset,
        )
        if prev_dep is not None and layer_ops:
            layer_ops[0].depends_on = [prev_dep]
        prefill_ops.extend(layer_ops)
        if layer_ops:
            prev_dep = len(prefill_ops) - 1

    # ------ Decode (per token) -------------------------------------------------
    decode_ops: List[SimOp] = []
    prev_dep = None

    for layer_cfg in stage_layers:
        offset = len(decode_ops)
        layer_ops = build_layer_ops(
            layer_cfg=layer_cfg,
            model_cfg=model_cfg,
            parallel_cfg=parallel_cfg,
            hw=hw,
            batch=batch,
            seq_len=1,
            phase=Phase.DECODE,
            kv_len=kv_len,
            index_offset=offset,
        )
        if prev_dep is not None and layer_ops:
            layer_ops[0].depends_on = [prev_dep]
        decode_ops.extend(layer_ops)
        if layer_ops:
            prev_dep = len(decode_ops) - 1

    return prefill_ops, decode_ops
