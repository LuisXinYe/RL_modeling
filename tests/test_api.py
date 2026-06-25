"""Tests for the FastAPI REST API."""
import pytest
from fastapi.testclient import TestClient
from llm_perf.ui.api import app

client = TestClient(app)

def test_index_returns_html():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "llm-perf" in r.text

def test_get_models():
    r = client.get("/api/models")
    assert r.status_code == 200
    templates = r.json()["templates"]
    assert "Llama-3.1-8B" in templates
    assert templates["Llama-3.1-8B"]["hidden_size"] == 4096

def test_get_hardware():
    r = client.get("/api/hardware")
    assert r.status_code == 200
    profiles = r.json()["profiles"]
    assert "Ascend 910C" in profiles
    assert profiles["Ascend 910C"]["devices_per_node"] == 8

def test_predict_default():
    r = client.post("/api/predict", json={})
    assert r.status_code == 200
    data = r.json()
    assert "kpis" in data
    assert "memory" in data
    assert "timeline" in data
    assert "topology" in data
    assert data["kpis"]["step_time_seconds"] > 0
    assert data["kpis"]["gen_tps_target"] > 0

def test_predict_custom_config():
    r = client.post("/api/predict", json={
        "model": {"name": "test", "hidden_size": 4096, "vocab_size": 32000, "num_layers": 16, "dtype": "bf16"},
        "hardware": "Ascend 910C",
        "total_devices": 16,
        "parallelism": {"tp": 2, "pp": 1, "dp": 8, "ep": 1, "cp": 1},
    })
    assert r.status_code == 200
    assert r.json()["kpis"]["step_time_seconds"] > 0

def test_predict_invalid_hardware():
    r = client.post("/api/predict", json={"hardware": "NonExistent"})
    assert r.status_code in (400, 422)

def test_search_pareto():
    r = client.post("/api/search", json={
        "search": {"mode": "pareto", "device_counts": [8, 16]},
    })
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    assert len(data["results"]) > 0
    assert "status" in data

def test_search_sensitivity():
    r = client.post("/api/search", json={
        "search": {"mode": "sensitivity", "sweep_param": "group_size", "sweep_values": [4, 8]},
    })
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2

def test_static_files():
    assert client.get("/static/styles.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200


# ── Scenario routing (inference / pretraining / post-training) ────────

def test_predict_post_training_scenario():
    r = client.post("/api/predict", json={"scenario": "post_training"})
    assert r.status_code == 200
    data = r.json()
    assert data["scenario"] == "post_training"
    k = data["kpis"]
    assert k["step_time_seconds"] > 0
    assert k["gen_tps_target"] > 0
    assert k["train_tps_target"] > 0
    assert k["ref_tps_target"] > 0


def test_predict_inference_scenario():
    r = client.post("/api/predict", json={"scenario": "inference"})
    assert r.status_code == 200
    data = r.json()
    assert data["scenario"] == "inference"
    k = data["kpis"]
    assert k["gen_tps_target"] > 0
    assert k["gen_time_seconds"] > 0
    # prefill + decode should account for total gen time
    assert k["prefill_seconds"] + k["decode_seconds"] == pytest.approx(
        k["gen_time_seconds"], rel=0.05
    )
    assert data["memory"]["kv_cache_gb"] > 0
    assert "total_gen_gb" in data["memory"]
    # inference response must NOT carry training/ref fields
    assert "train_tps_target" not in k
    assert "ref_tps_target" not in k


def test_predict_pretraining_scenario():
    r = client.post("/api/predict", json={"scenario": "pretraining"})
    assert r.status_code == 200
    data = r.json()
    assert data["scenario"] == "pretraining"
    k = data["kpis"]
    assert k["step_time_seconds"] > 0
    assert k["train_tps_target"] > 0
    assert k["total_train_gb"] > 0
    # pretraining has no generation or reference phases
    assert "gen_tps_target" not in k
    assert "ref_tps_target" not in k
    bd = data["timeline"]["breakdown"]
    assert bd["total"] == pytest.approx(k["step_time_seconds"], rel=0.05)
    assert {"weight_gb", "optimizer_gb", "activation_peak_gb"} <= set(
        data["memory"]
    )


def test_predict_pretraining_excludes_rl_substeps():
    """Pretraining step time must be < post-training step time for the same
    config (no reward_fwd / old_logprob_fwd, no gen, no ref phases)."""
    pre = client.post("/api/predict", json={"scenario": "pretraining"}).json()
    post = client.post("/api/predict", json={"scenario": "post_training"}).json()
    assert pre["kpis"]["step_time_seconds"] < post["kpis"]["step_time_seconds"]
