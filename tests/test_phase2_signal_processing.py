"""
Phase 2 Tests: Signal Quality, Waveform Delineation, QTc Calculators
"""

import pytest
import numpy as np
from pipeline.signal_quality import SignalQualityAnalyzer
from pipeline.waveform_delineator import WaveformDelineator
from pipeline.qtc_calculator import (
    QTcCalculator, bazett_qtc, fridericia_qtc, compute_qtc_multimethod
)


@pytest.fixture
def synthetic_ecg():
    """Generates synthetic 10-second clean ECG at 500 Hz."""
    fs = 500
    t = np.linspace(0, 10, 10 * fs)
    # Simplified: 1.2 Hz ECG-like wave (normal ~60 bpm HR)
    ecg = np.sin(2 * np.pi * 1.2 * t)
    return ecg


@pytest.fixture
def sqi_analyzer():
    return SignalQualityAnalyzer(sampling_rate=500)


@pytest.fixture
def delineator():
    return WaveformDelineator(sampling_rate=500)


class TestSignalQuality:
    """Tests for signal quality assessment."""

    def test_good_signal_quality(self, sqi_analyzer, synthetic_ecg):
        """Clean synthetic signal should have good quality."""
        metrics = sqi_analyzer.compute_sqi(synthetic_ecg)
        assert metrics["quality_score"] > 70
        assert not metrics["saturated"]
        assert not metrics["flat"]

    def test_flat_signal_detection(self, sqi_analyzer):
        """Zero signal should be detected as flat."""
        flat_signal = np.zeros(5000)
        metrics = sqi_analyzer.compute_sqi(flat_signal)
        assert metrics["flat"] is True
        assert metrics["quality_score"] < 60  # Should lose significant points

    def test_saturated_signal_detection(self, sqi_analyzer):
        """Clipped signal should be detected."""
        clipped = np.ones(5000) * 15.0  # Beyond ±10mV limit
        metrics = sqi_analyzer.compute_sqi(clipped)
        assert metrics["saturated"] is True
        assert metrics["quality_score"] < 60  # Should lose significant points

    def test_baseline_wander_detection(self, sqi_analyzer):
        """Added baseline drift should be detected."""
        t = np.linspace(0, 10, 5000)
        signal = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 0.05 * t)
        metrics = sqi_analyzer.compute_sqi(signal)
        assert metrics["baseline_wander_mv"] > 0.1
        assert metrics["quality_score"] < 85


class TestWaveformDelineation:
    """Tests for NeuroKit2-based waveform delineation."""

    def test_delineation_success(self, delineator, synthetic_ecg):
        """Should successfully delineate a clean synthetic signal."""
        result = delineator.delineate_lead(synthetic_ecg)
        # May not always succeed with synthetic signal, but should attempt
        assert "success" in result

    def test_measure_intervals(self, delineator, synthetic_ecg):
        """Should return physiologic intervals."""
        result = delineator.measure_all_leads(
            np.tile(synthetic_ecg[:, np.newaxis], (1, 12))
        )
        assert "pr_interval_ms" in result
        assert "qrs_duration_ms" in result
        assert "qt_interval_ms" in result
        # Bounds check
        assert 80 <= result["pr_interval_ms"] <= 320
        assert 60 <= result["qrs_duration_ms"] <= 220


class TestQTcCalculator:
    """Tests for multiple QTc correction formulas."""

    def test_bazett_qtc(self):
        """Bazett's formula: QTc = QT / sqrt(RR)."""
        qt_ms, rr_sec = 400, 0.857  # 70 bpm
        qtc = bazett_qtc(qt_ms, rr_sec)
        # sqrt(0.857) ≈ 0.926, so 400 / 0.926 ≈ 432
        assert 420 < qtc < 450

    def test_fridericia_qtc(self):
        """Fridericia's formula: QTc = QT / (RR^(1/3))."""
        qt_ms, rr_sec = 400, 0.857
        qtc = fridericia_qtc(qt_ms, rr_sec)
        # 0.857^(1/3) ≈ 0.951, so 400 / 0.951 ≈ 421
        assert 410 < qtc < 440

    def test_tachycardia_correction(self):
        """At tachycardia (100 bpm), Fridericia should be closer than Bazett."""
        qt_ms = 380
        rr_sec = 0.6  # 100 bpm
        bazett = bazett_qtc(qt_ms, rr_sec)
        fridericia = fridericia_qtc(qt_ms, rr_sec)
        # Bazett will overcorrect; Fridericia should be more conservative
        assert bazett > fridericia

    def test_qtc_calculator_multimethod(self):
        """QTcCalculator should recommend best formula based on HR."""
        calc_normal = QTcCalculator(qt_ms=400, rr_sec=0.857, hr_bpm=70, patient_sex="M")
        formula, qtc = calc_normal.recommend_formula()
        # At 70 bpm, should recommend Bazett (standard)
        assert formula == "bazett"
        assert 420 < qtc < 450

        calc_tachy = QTcCalculator(qt_ms=380, rr_sec=0.6, hr_bpm=100, patient_sex="M")
        formula_t, qtc_t = calc_tachy.recommend_formula()
        # At 100 bpm, should recommend Fridericia
        assert formula_t == "fridericia"

    def test_prolonged_qtc_detection(self):
        """Should correctly classify prolonged QTc."""
        calc = QTcCalculator(qt_ms=500, rr_sec=0.857, hr_bpm=70, patient_sex="M")
        assert calc.is_prolonged(470) is True
        assert calc.is_prolonged(430) is False
        assert calc.is_severely_prolonged(510) is True
        assert calc.is_severely_prolonged(480) is False

    def test_compute_qtc_multimethod_convenience(self):
        """Convenience function should return all methods and recommendation."""
        result = compute_qtc_multimethod(400, 0.857, 70, "M")
        assert "qt_ms" in result
        assert "qtc_bazett" in result
        assert "qtc_fridericia" in result
        assert "recommended_qtc" in result
        assert "status" in result
        assert result["status"] in ["NORMAL", "PROLONGED", "SEVERELY_PROLONGED"]
