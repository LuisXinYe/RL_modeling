import pytest
from llm_perf.precision import (
    dtype_bytes, compute_class, scale_overhead_bytes,
    TensorPrecision, PrecisionConfig,
)


def test_dtype_bytes_map():
    assert dtype_bytes("fp32") == 4
    assert dtype_bytes("bf16") == 2
    assert dtype_bytes("fp16") == 2
    assert dtype_bytes("fp8_e4m3") == 1
    assert dtype_bytes("fp8_e5m2") == 1
    assert dtype_bytes("fp4_e2m1") == 0.5
    assert dtype_bytes("mxfp4") == 0.5


def test_compute_class_map():
    assert compute_class("bf16") == "bf16"
    assert compute_class("fp16") == "bf16"
    assert compute_class("fp32") == "bf16"
    assert compute_class("fp8_e4m3") == "fp8"
    assert compute_class("fp8_e5m2") == "fp8"
    assert compute_class("fp4_e2m1") == "fp4"
    assert compute_class("mxfp4") == "fp4"


def test_unknown_dtype_raises():
    with pytest.raises(KeyError):
        dtype_bytes("int3")


def test_scale_overhead_per_tensor_is_negligible():
    # block_size=0 → one scale per tensor
    assert scale_overhead_bytes(1024, block_size=0, scale_bytes=4) == 4


def test_scale_overhead_fine_grained():
    # 1024 elements, block 128 → 8 scales * 4 bytes
    assert scale_overhead_bytes(1024, block_size=128, scale_bytes=4) == 32


def test_precision_config_default_is_all_bf16():
    cfg = PrecisionConfig.bf16_default()
    assert cfg.weights.dtype == "bf16"
    assert cfg.activations.dtype == "bf16"
    assert cfg.gradients.dtype == "bf16"
    assert cfg.comm.dtype == "bf16"
    assert cfg.master_dtype == "fp32"


def test_tensor_precision_fields():
    tp = TensorPrecision(dtype="fp4_e2m1", block_size=128, hadamard=True, hadamard_block=128)
    assert tp.block_size == 128
    assert tp.hadamard is True
