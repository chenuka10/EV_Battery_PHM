import io
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_api_health():
    """Verify health endpoint returns status and artifact flags."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["artifacts"]["champion_rul_hgb"] is True
    assert data["artifacts"]["champion_failure_lr"] is True
    assert "fastapi" in data["library_versions"]


def test_api_meta():
    """Verify metadata endpoint returns schema and physical constants."""
    res = client.get("/api/meta")
    assert res.status_code == 200
    data = res.json()
    assert len(data["features"]) == 73  # 66 base + 7 engineered features transformed by CT
    assert data["constants"]["tau_safety"] == 0.193
    assert data["constants"]["test_set_size"] == 4000
    assert "engineered_formulas" in data


def test_api_predict_happy_paths():
    """Verify predict endpoint executes cleanly for both engines."""
    meta_res = client.get("/api/meta").json()
    preset1 = meta_res["scenarios"]["Scenario 1: Factory-Fresh Battery (Low Cycles, Peak Health)"]["data"]

    # Logistic engine
    res_lr = client.post("/api/predict", json={
        "engine": "logistic",
        "tau": 0.193,
        "inputs": preset1
    })
    assert res_lr.status_code == 200
    d_lr = res_lr.json()
    assert d_lr["rul_cycles"] > 5000
    assert 0.0 <= d_lr["p_failure"] <= 1.0
    assert d_lr["status"] == "nominal"

    # XGBoost engine
    res_xgb = client.post("/api/predict", json={
        "engine": "xgboost",
        "tau": 0.193,
        "inputs": preset1
    })
    assert res_xgb.status_code == 200
    d_xgb = res_xgb.json()
    assert 0.0 <= d_xgb["p_failure"] <= 1.0


def test_api_predict_validation_error():
    """Verify that impossible physical inputs return 422 with structured error."""
    bad_req = {
        "engine": "logistic",
        "tau": 0.193,
        "inputs": {
            "cell_temperature_max": 20.0,
            "cell_temperature_avg": 35.0  # Anomaly: Max < Avg
        }
    }
    res = client.post("/api/predict", json=bad_req)
    assert res.status_code == 422
    data = res.json()
    assert "Physical Anomaly" in data["detail"]


def test_api_threshold():
    """Verify threshold query endpoint."""
    res = client.get("/api/threshold?engine=logistic&tau=0.193")
    assert res.status_code == 200
    data = res.json()
    assert "metrics" in data
    assert "curve" in data
    assert data["metrics"]["recall"] >= 0.89


def test_api_modes():
    """Verify operating modes endpoint."""
    res = client.get("/api/modes?engine=logistic")
    assert res.status_code == 200
    data = res.json()
    assert "safety" in data["modes"]
    assert "balanced" in data["modes"]


def test_api_simulate():
    """Verify fleet decision simulation endpoint."""
    res = client.post("/api/simulate", json={
        "n_vehicles": 5000,
        "daily_capacity": 300,
        "engine": "logistic",
        "tau": 0.193
    })
    assert res.status_code == 200
    data = res.json()
    assert data["fleet_size"] == 5000
    assert "expected_flags" in data


def test_api_explain():
    """Verify attribution endpoint."""
    meta_res = client.get("/api/meta").json()
    preset3 = meta_res["scenarios"]["Scenario 3: Incipient Thermal Hazard (Passes at 0.50, INTERCEPTED at 0.193!)"]["data"]

    res = client.post("/api/explain", json={
        "engine": "logistic",
        "inputs": preset3
    })
    assert res.status_code == 200
    data = res.json()
    assert len(data["top_risk_factors"]) > 0
    assert "caption" in data


def test_api_sample_batch_and_batch_upload():
    """Verify sample batch download and subsequent upload to batch processor."""
    # 1. Download sample CSV
    sample_res = client.get("/api/sample-batch")
    assert sample_res.status_code == 200
    csv_bytes = sample_res.content
    assert len(csv_bytes) > 100

    # 2. Upload to /api/batch
    files = {"file": ("test_batch.csv", io.BytesIO(csv_bytes), "text/csv")}
    batch_res = client.post("/api/batch?engine=logistic&tau=0.193", files=files)
    assert batch_res.status_code == 200
    b_data = batch_res.json()
    assert b_data["total_records"] == 4
    assert len(b_data["ranked_records"]) == 4
    assert "csv_download" in b_data


def test_api_leaderboard():
    """Verify master leaderboards reproduce report figures."""
    res = client.get("/api/leaderboard")
    assert res.status_code == 200
    data = res.json()
    assert len(data["task1"]) == 7
    assert len(data["task2"]) == 6
    # Verify Task 1 champion
    hgb = next(r for r in data["task1"] if "HistGradientBoosting" in r["model"])
    assert hgb["r2"] == 0.8972
    assert hgb["rmse"] == 528.64
    # Verify Task 2 champions
    xgb = next(r for r in data["task2"] if "Tuned XGBoost" in r["model"])
    assert xgb["recall"] == "91.70%"
    assert xgb["pr_auc"] == 0.7804


def test_api_data_health():
    """Verify data health facts match Technical Report Section 3."""
    res = client.get("/api/data-health")
    assert res.status_code == 200
    data = res.json()
    assert data["total_records"] == 20000
    assert data["incomplete_records"] == 18663
    assert data["complete_records"] == 1337
    assert data["excluded_task1_records"] == 898


def test_api_selftest():
    """Verify automated self-test runs all presets and all pass."""
    res = client.get("/api/selftest")
    assert res.status_code == 200
    data = res.json()
    assert data["all_passed"] is True
    assert len(data["results"]) == 4
