"""
Unit & Integration Tests for Ollama EKG Clinical Interpretation Client
"""

import pytest
from pipeline.ollama_client import OllamaEKGClient, MANDATORY_DISCLAIMER


@pytest.fixture(scope="module")
def client():
    return OllamaEKGClient(timeout=60.0)


@pytest.fixture
def sample_data():
    features = {
        "heart_rate_bpm": 82.5,
        "rhythm_regular": True,
        "pr_interval_ms": 160.0,
        "qrs_duration_ms": 85.0,
        "qt_interval_ms": 360.0,
        "qtc_interval_ms": 421.8,
        "heart_axis_deg": 45.0,
        "heart_axis_category": "Normal Axis",
        "sokolow_lyon_mv": 2.2,
        "lvh_voltage_criteria": False,
        "st_deviations_mv": {"V1": 0.0, "V2": 0.0, "V3": 0.0},
        "st_elevation_leads": [],
        "st_depression_leads": [],
        "stemi_suspected": False,
        "wide_qrs": False,
        "prolonged_pr": False,
        "prolonged_qtc": False,
    }
    classification = {
        "top_findings": [("NORM", 0.92), ("STTC", 0.05), ("MI", 0.02)],
        "primary_findings": [],
        "probabilities": {"MI": 0.02, "NORM": 0.92},
        "clinical_urgency": "ROUTINE",
    }
    patient = {"age": 55, "sex": "M", "symptoms": "건강검진"}
    return features, classification, patient


def test_ollama_health(client):
    """Verifies that the local Ollama server is responding."""
    assert client.check_health() is True, "Ollama service is not reachable on localhost:11434"


def test_prompt_construction(client, sample_data):
    """Verifies prompt composition includes clinical intervals and deep learning probabilities."""
    features, classification, patient = sample_data
    prompt = client.build_clinical_prompt(features, classification, patient)

    assert "55세" in prompt
    assert "남성" in prompt
    assert "82.5 bpm" in prompt
    assert "160.0 ms" in prompt
    assert "421.8 ms" in prompt
    assert "NORM (92.0%)" in prompt
    assert "ROUTINE" in prompt


def test_mandatory_disclaimer_constant():
    """Verifies standard legal disclaimer text."""
    assert "본 판독문은 인공지능" in MANDATORY_DISCLAIMER
    assert "최종 진단" in MANDATORY_DISCLAIMER
    assert "담당 의사" in MANDATORY_DISCLAIMER


def test_generate_structured_json_live(client, sample_data):
    """Integration test: calls Ollama to produce a validated JSON payload."""
    features, classification, patient = sample_data
    result = client.generate_structured_json(features, classification, patient)

    assert isinstance(result, dict)
    assert "disclaimer" in result
    assert MANDATORY_DISCLAIMER in result["disclaimer"]

    # Verify key clinical fields
    if "error" not in result:
        assert "rhythm" in result or "primary_diagnosis" in result
        assert "urgency" in result
        assert result["urgency"] in ["EMERGENCY", "URGENT", "ROUTINE"]
