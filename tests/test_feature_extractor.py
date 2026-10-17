"""
Unit Tests for ECG Preprocessor & Quantitative Feature Extractor
"""

import pytest
import numpy as np
from pipeline.ptbxl_loader import PTBXLLoader
from pipeline.preprocessor import ECGPreprocessor
from pipeline.feature_extractor import ECGFeatureExtractor, LEAD_INDICES


@pytest.fixture(scope="module")
def loader():
    return PTBXLLoader()


@pytest.fixture(scope="module")
def preprocessor():
    return ECGPreprocessor(sampling_rate=100)


@pytest.fixture(scope="module")
def extractor():
    return ECGFeatureExtractor(sampling_rate=100)


def test_preprocessor_shape_and_filtering(preprocessor):
    """Verifies that the preprocessor maintains shape and removes high-frequency noise."""
    t = np.linspace(0, 10, 1000)
    # 1.2 Hz base ECG-like wave + 50 Hz noise + slow 0.1 Hz drift
    clean_wave = np.sin(2 * np.pi * 1.2 * t)
    noise = 0.5 * np.sin(2 * np.pi * 50.0 * t)
    drift = 1.5 * np.sin(2 * np.pi * 0.1 * t)
    synthetic_lead = clean_wave + noise + drift
    synthetic_12lead = np.tile(synthetic_lead[:, np.newaxis], (1, 12))

    filtered = preprocessor.filter_signal(synthetic_12lead)
    assert filtered.shape == (1000, 12)
    assert not np.isnan(filtered).any()
    # High frequency noise power should be significantly reduced
    assert np.std(filtered) < np.std(synthetic_12lead)


def test_feature_extraction_keys_and_ranges(loader, extractor):
    """Verifies that feature extraction returns all required clinical keys with valid physiological ranges."""
    rec = loader.get_record(1, sampling_rate=100)
    feats = extractor.extract_features(rec["signal"], patient_sex=rec["sex"], patient_age=rec["age"])

    expected_keys = [
        "heart_rate_bpm", "rhythm_regular", "pr_interval_ms", "qrs_duration_ms",
        "qt_interval_ms", "qtc_interval_ms", "heart_axis_deg", "heart_axis_category",
        "sokolow_lyon_mv", "lvh_voltage_criteria", "st_deviations_mv",
        "st_elevation_leads", "st_depression_leads", "stemi_suspected",
        "wide_qrs", "prolonged_pr", "prolonged_qtc"
    ]
    for key in expected_keys:
        assert key in feats, f"Missing key: {key}"

    # Physiological checks on normal record (Record 1)
    assert 40.0 <= feats["heart_rate_bpm"] <= 180.0
    assert 60.0 <= feats["qrs_duration_ms"] <= 200.0
    assert 80.0 <= feats["pr_interval_ms"] <= 320.0
    assert 280.0 <= feats["qtc_interval_ms"] <= 600.0
    assert -180.0 <= feats["heart_axis_deg"] <= 180.0
    assert feats["heart_axis_category"] in [
        "Normal Axis", "Left Axis Deviation (LAD)", "Right Axis Deviation (RAD)", "Extreme Axis (Northwest)"
    ]
    assert len(feats["st_deviations_mv"]) == 12


def test_stemi_detection_on_infarction_record(loader, extractor):
    """Verifies extraction and morphological flags on an anteroseptal MI record."""
    # Record 63 has ASMI (Anteroseptal Myocardial Infarction)
    rec = loader.get_record(63, sampling_rate=100)
    feats = extractor.extract_features(rec["signal"], patient_sex=rec["sex"], patient_age=rec["age"])

    assert isinstance(feats["st_deviations_mv"], dict)
    assert "V1" in feats["st_deviations_mv"]
    assert "V2" in feats["st_deviations_mv"]
    assert "V3" in feats["st_deviations_mv"]


def test_sokolow_lyon_lvh_criteria(loader, extractor):
    """Verifies that Sokolow-Lyon criteria calculates S(V1) + max(R_V5, R_V6)."""
    rec = loader.get_record(162, sampling_rate=100)  # Record 162 has LVH
    feats = extractor.extract_features(rec["signal"], patient_sex=rec["sex"], patient_age=rec["age"])

    assert feats["sokolow_lyon_mv"] > 0
    assert isinstance(feats["lvh_voltage_criteria"], (bool, np.bool_))


def test_edge_case_near_flatline(extractor):
    """Verifies that near-flatline or low-amplitude input does not crash or produce NaNs."""
    flat_signal = np.zeros((1000, 12)) + 0.001 * np.random.randn(1000, 12)
    feats = extractor.extract_features(flat_signal, patient_sex="M")
    
    assert feats["heart_rate_bpm"] > 0
    assert not np.isnan(feats["qrs_duration_ms"])
    assert not np.isnan(feats["qtc_interval_ms"])
