"""Tests for the FastAPI REST API."""
import pytest
from fastapi.testclient import TestClient
from rl_perf.ui.api import app

client = TestClient(app)

def test_index_returns_html():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "rl-perf" in r.text

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
    assert data["kpis"]["epoch_time_hours"] > 0
    assert data["kpis"]["bottleneck"].lower() in ("generation", "training")

def test_predict_custom_config():
    r = client.post("/api/predict", json={
        "model": {"name": "test", "hidden_size": 4096, "vocab_size": 32000, "num_layers": 16, "dtype": "bf16"},
        "hardware": "Ascend 910C",
        "total_devices": 16,
        "parallelism": {"tp": 2, "pp": 1, "dp": 8, "ep": 1, "cp": 1},
    })
    assert r.status_code == 200
    assert r.json()["kpis"]["epoch_time_hours"] > 0

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
