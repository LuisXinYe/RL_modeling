"""Per-tensor-role precision resolution for low-precision training modeling.

Pure functions + pydantic config. No dependency on builder/simulator/ops so it
can be unit-tested in isolation. See
docs/superpowers/specs/2026-06-27-low-precision-training-modeling-design.md.
"""

from __future__ import annotations

from pydantic import BaseModel

# Single source of truth for element sizes. fp4 stores two values per byte.
# Ref: NVIDIA OCP MX spec (mxfp4 = 4-bit element + shared e8m0 scale);
# FP8 formats per "FP8 Formats for Deep Learning" (Micikevicius et al. 2022).
_DTYPE_BYTES = {
    "fp32": 4,
    "bf16": 2,
    "fp16": 2,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
    "fp4_e2m1": 0.5,
    "mxfp4": 0.5,
}

_COMPUTE_CLASS = {
    "fp32": "bf16",
    "bf16": "bf16",
    "fp16": "bf16",
    "fp8_e4m3": "fp8",
    "fp8_e5m2": "fp8",
    "fp4_e2m1": "fp4",
    "mxfp4": "fp4",
}


def dtype_bytes(dtype: str) -> float:
    """Bytes per element for a dtype (may be fractional, e.g. fp4 = 0.5)."""
    return _DTYPE_BYTES[dtype]


def compute_class(dtype: str) -> str:
    """Map a dtype to its matmul compute pipe class: 'bf16' | 'fp8' | 'fp4'."""
    return _COMPUTE_CLASS[dtype]


def scale_overhead_bytes(numel: float, block_size: int, scale_bytes: int = 4) -> float:
    """Quantization scale-metadata bytes.

    block_size=0 → one scale per tensor. block_size=B → ceil(numel/B) scales.
    Ref: fine-grained block scaling (DeepSeek-V3 tech report, per-128 blocks).
    """
    if block_size <= 0:
        return float(scale_bytes)
    num_blocks = -(-int(numel) // block_size)  # ceil division
    return float(num_blocks * scale_bytes)


class TensorPrecision(BaseModel):
    dtype: str = "bf16"
    block_size: int = 0          # 0 = per-tensor scale; >0 = fine-grained block
    hadamard: bool = False       # stochastic Hadamard transform before quant
    hadamard_block: int = 0      # rotation size; 0 → resolver default (128)
    scale_bytes: int = 4         # bytes per scale (4=fp32, 1=e8m0 for mxfp4)


class PrecisionConfig(BaseModel):
    weights: TensorPrecision = TensorPrecision()
    activations: TensorPrecision = TensorPrecision()
    gradients: TensorPrecision = TensorPrecision()
    comm: TensorPrecision = TensorPrecision()
    master_dtype: str = "fp32"
    error_feedback: bool = False
    ef_dtype: str = "fp16"
    high_precision_layers: list[str] = []
    high_precision_period: int = 0
    high_precision_dtype: str = "bf16"

    @classmethod
    def bf16_default(cls) -> "PrecisionConfig":
        """All-bf16 recipe — reproduces today's (single-dtype) behavior."""
        return cls()
