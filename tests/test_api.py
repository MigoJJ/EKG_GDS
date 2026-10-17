"""
Integration Tests for FastAPI EKG AI Reading Assistant Service
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_api_health(client):
    """Verifies API health check and component statuses."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert data["onnx_model_loaded"] is True
    assert data["version"] == "1.0.0"


def test_extract_features_endpoint(client):
    """Verifies /api/v1/ecg/features endpoint with synthetic 12-lead signal."""
    # 1000 samples x 12 leads
    dummy_signal = (0.5 * np.sin(np.linspace(0, 20, 1000)[:, np.newaxis]) * np.ones((1, 12))).tolist()
    payload = {
        "signal": dummy_signal,
        "sampling_rate": 100,
        "patient_age": 45,
        "patient_sex": "M"
    }

    response = client.post("/api/v1/ecg/features", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "heart_rate_bpm" in data
    assert "pr_interval_ms" in data
    assert "qrs_duration_ms" in data
    assert "qtc_interval_ms" in data
    assert "heart_axis_deg" in data
    assert "sokolow_lyon_mv" in data


def test_classify_endpoint(client):
    """Verifies /api/v1/ecg/classify endpoint with synthetic signal."""
    dummy_signal = np.random.randn(1000, 12).astype(np.float32).tolist()
    payload = {
        "signal": dummy_signal,
        "sampling_rate": 100
    }

    response = client.post("/api/v1/ecg/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "probabilities" in data
    assert len(data["probabilities"]) == 11
    assert "clinical_urgency" in data
    assert "top_findings" in data


def test_analyze_sample_ptbxl_record(client):
    """Verifies /api/v1/ecg/sample/{ecg_id} endpoint with a real PTB-XL record."""
    response = client.get("/api/v1/ecg/sample/9")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert "patient_info" in data
    assert data["patient_info"]["age"] == 55.0
    assert "quantitative_features" in data
    assert "deep_learning_classification" in data
    assert "latency_ms" in data
    assert "disclaimer" in data
    assert "본 판독문은 인공지능" in data["disclaimer"]


def test_invalid_input_error_handling(client):
    """Verifies validation errors for bad input shapes and missing fields."""
    # Empty payload
    res1 = client.post("/api/v1/ecg/analyze", json={})
    assert res1.status_code == 400

    # Non-existent record ID
    res2 = client.post("/api/v1/ecg/analyze", json={"ecg_id": 999999})
    assert res2.status_code == 404
