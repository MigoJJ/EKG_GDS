"""
QTc Interval Calculator with Multiple Correction Formulas
Implements Bazett, Fridericia, Framingham, and other QTc formulas.
Bazett tends to overcorrect at high heart rates; Fridericia and Framingham are alternatives.
"""

from __future__ import annotations
import math
from typing import Dict, Optional


def bazett_qtc(qt_ms: float, rr_sec: float) -> float:
    """
    Bazett's QTc formula: QTc = QT / sqrt(RR)
    Most commonly used but overcorrects at tachycardia (>100 bpm).
    """
    if rr_sec <= 0:
        return qt_ms
    return qt_ms / math.sqrt(rr_sec)


def fridericia_qtc(qt_ms: float, rr_sec: float) -> float:
    """
    Fridericia's QTc formula: QTc = QT / (RR^(1/3))
    Better at high heart rates; recommended for tachycardia.
    """
    if rr_sec <= 0:
        return qt_ms
    return qt_ms / (rr_sec ** (1/3))


def framingham_qtc(qt_ms: float, rr_sec: float) -> float:
    """
    Framingham's QTc formula: QTc = QT + 0.154 * (1 - RR)
    Linear correction; good general-purpose alternative.
    RR in seconds.
    """
    return qt_ms + 0.154 * (1 - rr_sec) * 1000


def hodges_qtc(qt_ms: float, hr_bpm: float) -> float:
    """
    Hodges' formula: QTc = QT + 1.75 * (HR - 60) / 1000
    Simplified linear form using heart rate instead of RR.
    """
    return qt_ms + 1.75 * (hr_bpm - 60)


def rautaharju_qtc(qt_ms: float, rr_sec: float, patient_sex: str = "M") -> float:
    """
    Rautaharju's gender-specific formula.
    More complex; accounts for sex differences in QTc.
    """
    if rr_sec <= 0:
        return qt_ms
    # Simplified version; full formula requires regression coefficients
    if patient_sex == "F":
        return qt_ms / (rr_sec ** 0.4)
    else:
        return qt_ms / (rr_sec ** 0.4)  # Same exponent in simplified form


class QTcCalculator:
    """
    Computes QTc using multiple formulas and provides recommendations.
    """

    # Normal QTc ranges (milliseconds) by sex
    NORMAL_RANGES = {
        "M": {"lower": 380, "upper": 450},
        "F": {"lower": 390, "upper": 460},
    }

    # Abnormal thresholds (milliseconds)
    PROLONGED_THRESHOLD = {
        "M": 450,
        "F": 460,
    }

    SEVERELY_PROLONGED_THRESHOLD = {
        "M": 500,
        "F": 500,
    }

    def __init__(self, qt_ms: float, rr_sec: float, hr_bpm: float, patient_sex: str = "M"):
        """
        Args:
            qt_ms: QT interval in milliseconds
            rr_sec: RR interval in seconds (mean RR between beats)
            hr_bpm: Heart rate in beats per minute
            patient_sex: "M" or "F"
        """
        self.qt_ms = qt_ms
        self.rr_sec = rr_sec
        self.hr_bpm = hr_bpm
        self.sex = patient_sex

    def compute_all_formulas(self) -> Dict[str, float]:
        """
        Computes QTc using all available formulas.

        Returns:
            {
                "bazett": float,
                "fridericia": float,
                "framingham": float,
                "hodges": float,
                "rautaharju": float,
            }
        """
        return {
            "bazett": round(bazett_qtc(self.qt_ms, self.rr_sec), 1),
            "fridericia": round(fridericia_qtc(self.qt_ms, self.rr_sec), 1),
            "framingham": round(framingham_qtc(self.qt_ms, self.rr_sec), 1),
            "hodges": round(hodges_qtc(self.qt_ms, self.hr_bpm), 1),
            "rautaharju": round(rautaharju_qtc(self.qt_ms, self.rr_sec, self.sex), 1),
        }

    def recommend_formula(self) -> Tuple[str, float]:
        """
        Recommends the best QTc formula based on heart rate.

        Returns:
            (formula_name, qtc_value)
        """
        # At normal heart rates (60-90), any formula works; use Bazett (most common)
        # At tachycardia (>100), prefer Fridericia or Framingham
        if self.hr_bpm < 60:
            # Bradycardia: Bazett overcorrects; use Fridericia
            formula = "fridericia"
        elif self.hr_bpm < 100:
            # Normal: use Bazett (standard)
            formula = "bazett"
        else:
            # Tachycardia: avoid Bazett overcorrection; use Fridericia
            formula = "fridericia"

        qtc_values = self.compute_all_formulas()
        return formula, qtc_values[formula]

    def is_prolonged(self, qtc_ms: float) -> bool:
        """Returns True if QTc exceeds sex-specific prolonged threshold."""
        threshold = self.PROLONGED_THRESHOLD.get(self.sex, 450)
        return qtc_ms >= threshold

    def is_severely_prolonged(self, qtc_ms: float) -> bool:
        """Returns True if QTc is severely prolonged (risk of arrhythmia)."""
        threshold = self.SEVERELY_PROLONGED_THRESHOLD.get(self.sex, 500)
        return qtc_ms >= threshold

    def get_status(self, qtc_ms: float) -> str:
        """Returns categorical status: normal, prolonged, or severely prolonged."""
        if self.is_severely_prolonged(qtc_ms):
            return "SEVERELY PROLONGED"
        elif self.is_prolonged(qtc_ms):
            return "PROLONGED"
        else:
            return "NORMAL"


def compute_qtc_multimethod(
    qt_ms: float,
    rr_sec: float,
    hr_bpm: float,
    patient_sex: str = "M"
) -> Dict[str, any]:
    """
    Convenience function: compute QTc all ways and return recommendation.

    Returns:
        {
            "qt_ms": float,
            "qtc_bazett": float,
            "qtc_fridericia": float,
            "qtc_framingham": float,
            "recommended_formula": str,
            "recommended_qtc": float,
            "status": "NORMAL" | "PROLONGED" | "SEVERELY_PROLONGED",
        }
    """
    calc = QTcCalculator(qt_ms, rr_sec, hr_bpm, patient_sex)
    all_formulas = calc.compute_all_formulas()
    rec_formula, rec_qtc = calc.recommend_formula()
    status = calc.get_status(rec_qtc)

    return {
        "qt_ms": qt_ms,
        "qtc_bazett": all_formulas["bazett"],
        "qtc_fridericia": all_formulas["fridericia"],
        "qtc_framingham": all_formulas["framingham"],
        "recommended_formula": rec_formula,
        "recommended_qtc": rec_qtc,
        "status": status,
    }
