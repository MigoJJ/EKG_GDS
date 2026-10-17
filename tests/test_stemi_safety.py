"""
STEMI Criteria Safety Regression Tests
Encodes the Fourth Universal Definition of MI ST-elevation thresholds:
- V2-V3: >=0.2 mV (men >=40), >=0.25 mV (men <40), >=0.15 mV (women)
- All other leads: >=0.1 mV
- Elevation required in >=2 anatomically contiguous leads

Tests marked xfail(strict=True) document known defects scheduled for Phase 1
(upgrade20261016.md). Once fixed they will XPASS and fail the suite, signalling
that the xfail marker should be removed.
"""

import pytest
import numpy as np
from pipeline.feature_extractor import ECGFeatureExtractor, LEAD_INDICES


@pytest.fixture
def extractor():
    return ECGFeatureExtractor(sampling_rate=100)


def _stemi_flag(extractor, monkeypatch, elevated: dict, sex: str, age=None) -> bool:
    """Runs feature extraction with ST deviations forced to the given per-lead values (mV)."""
    st = {lead: 0.0 for lead in LEAD_INDICES}
    st.update(elevated)
    monkeypatch.setattr(extractor, "_calculate_st_deviations", lambda filtered, peaks: dict(st))
    feats = extractor.extract_features(np.zeros((1000, 12)), patient_sex=sex, patient_age=age)
    return feats["stemi_suspected"]


def test_no_elevation_is_not_stemi(extractor, monkeypatch):
    assert _stemi_flag(extractor, monkeypatch, {}, sex="M", age=60) is False


def test_single_lead_elevation_is_not_stemi(extractor, monkeypatch):
    assert _stemi_flag(extractor, monkeypatch, {"II": 0.3}, sex="M", age=60) is False


def test_inferior_contiguous_elevation_is_stemi(extractor, monkeypatch):
    assert _stemi_flag(extractor, monkeypatch, {"II": 0.12, "III": 0.12}, sex="M", age=60) is True


def test_male_over_40_v2_v3_above_threshold_is_stemi(extractor, monkeypatch):
    assert _stemi_flag(extractor, monkeypatch, {"V2": 0.21, "V3": 0.21}, sex="M", age=60) is True


def test_female_v2_v3_above_threshold_is_stemi(extractor, monkeypatch):
    assert _stemi_flag(extractor, monkeypatch, {"V2": 0.16, "V3": 0.16}, sex="F", age=60) is True


def test_female_v2_v3_below_threshold_is_not_stemi(extractor, monkeypatch):
    """Female V2-V3 threshold correctly set to 0.15 mV (not 0.10 mV)."""
    assert _stemi_flag(extractor, monkeypatch, {"V2": 0.12, "V3": 0.12}, sex="F", age=60) is False


def test_male_under_40_v2_v3_below_threshold_is_not_stemi(extractor, monkeypatch):
    """Men <40 require >=0.25 mV in V2-V3 (not 0.20 mV), age is used."""
    assert _stemi_flag(extractor, monkeypatch, {"V2": 0.22, "V3": 0.22}, sex="M", age=35) is False


def test_non_contiguous_elevation_is_not_stemi(extractor, monkeypatch):
    """Non-contiguous elevation (III + V5) is not STEMI; requires anatomical contiguity."""
    assert _stemi_flag(extractor, monkeypatch, {"III": 0.12, "V5": 0.12}, sex="M", age=60) is False
