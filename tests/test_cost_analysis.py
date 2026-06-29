import pytest
from llm_perf.config import LayerConfig, ModelConfig
from llm_perf.precision import PrecisionConfig, ModuleLinearPrecision, TensorPrecision
from llm_perf.cost_analysis import theoretical_compute_cost, SPEED_MAP_PAPER


def llama7b_4k() -> ModelConfig:
    # MHA (num_kv_heads == num_heads), d=4096, d_ff=11008, 32 heads x 128, seq 4096
    layer = LayerConfig(
        attention="MHA", num_heads=32, num_kv_heads=32, head_dim=128,
        ffn="SwiGLU", intermediate_size=11008,
    )
    return ModelConfig(name="llama7b", hidden_size=4096, vocab_size=32000,
                       num_layers=32, default_layer=layer)


SEQ = 4096


def test_forward_split_matches_fig1a():
    pc = PrecisionConfig.bf16_default()
    out = theoretical_compute_cost(llama7b_4k(), pc, seq_len=SEQ)
    fs = out["forward_split"]
    assert fs["ffn"] == pytest.approx(0.573, abs=0.01)        # ~57%
    assert fs["attn_linear"] == pytest.approx(0.284, abs=0.01)  # ~28.7%
    assert fs["mha_core"] == pytest.approx(0.142, abs=0.01)     # ~14.3%
    assert sum(fs.values()) == pytest.approx(1.0, abs=1e-6)


def test_all_fp16_is_100pct():
    out = theoretical_compute_cost(llama7b_4k(), PrecisionConfig.bf16_default(), seq_len=SEQ)
    assert out["cost_pct"] == pytest.approx(100.0, abs=1e-6)


def test_all_fp4_under_stated_speeds_is_flop_honest_not_paper():
    # Documented finding: paper's stated FP4=4x yields ~36% for all-FP4, NOT 57.1%.
    all_fp4 = PrecisionConfig(
        attn_linear=ModuleLinearPrecision(fwd=TensorPrecision(dtype="fp4_e2m1"),
                                          bwd=TensorPrecision(dtype="fp4_e2m1")),
        ffn_linear=ModuleLinearPrecision(fwd=TensorPrecision(dtype="fp4_e2m1"),
                                         bwd=TensorPrecision(dtype="fp4_e2m1")),
    )
    out = theoretical_compute_cost(llama7b_4k(), all_fp4, seq_len=SEQ)
    assert out["cost_pct"] == pytest.approx(35.7, abs=2.0)  # FLOP-honest


def test_all_fp4_under_paper_implied_speeds_matches_57():
    all_fp4 = PrecisionConfig(
        attn_linear=ModuleLinearPrecision(fwd=TensorPrecision(dtype="fp4_e2m1"),
                                          bwd=TensorPrecision(dtype="fp4_e2m1")),
        ffn_linear=ModuleLinearPrecision(fwd=TensorPrecision(dtype="fp4_e2m1"),
                                         bwd=TensorPrecision(dtype="fp4_e2m1")),
    )
    out = theoretical_compute_cost(llama7b_4k(), all_fp4, seq_len=SEQ,
                                   speed_map={"fp16": 1.0, "fp8": 1.4, "fp4": 2.0})
    assert out["cost_pct"] == pytest.approx(57.1, abs=3.0)


def test_table2_ordering_attn_fp8_ffn_fp4_cheaper_than_attn_fp4_ffn_fp8():
    def recipe(attn, ffn):
        return PrecisionConfig(
            attn_linear=ModuleLinearPrecision(fwd=TensorPrecision(dtype=attn),
                                              bwd=TensorPrecision(dtype="fp4_e2m1")),
            ffn_linear=ModuleLinearPrecision(fwd=TensorPrecision(dtype=ffn),
                                             bwd=TensorPrecision(dtype="fp4_e2m1")),
        )
    m = llama7b_4k()
    c_attn8_ffn4 = theoretical_compute_cost(m, recipe("fp8_e4m3", "fp4_e2m1"), seq_len=SEQ)["cost_pct"]
    c_attn4_ffn8 = theoretical_compute_cost(m, recipe("fp4_e2m1", "fp8_e4m3"), seq_len=SEQ)["cost_pct"]
    # FFN is the bigger matmul, so quantizing it harder (FP4) is cheaper overall
    assert c_attn8_ffn4 < c_attn4_ffn8
