"""
Signal Quality Index (SQI) and Advanced Signal Processing
Detects noise, baseline wander, electrode disconnection, and other quality issues.
"""

from __future__ import annotations
import numpy as np
from scipy import signal as sp_signal


class SignalQualityAnalyzer:
    """Analyzes ECG signal quality to flag unreliable measurements."""

    def __init__(self, sampling_rate: int = 100):
        self.fs = sampling_rate

    def compute_sqi(self, lead_signal: np.ndarray) -> dict:
        """
        Computes signal quality metrics for a single ECG lead.

        Returns:
            {
                "snr_db": float,  # Signal-to-noise ratio (dB)
                "baseline_wander_mv": float,  # Max baseline wander (mV)
                "saturated": bool,  # True if signal is clipped
                "flat": bool,  # True if signal is flat/dead
                "quality_score": float,  # 0-100, 100=best
            }
        """
        metrics = {}

        # 1. Check for saturation (clipped signal)
        abs_max = np.max(np.abs(lead_signal))
        metrics["saturated"] = bool(abs_max > 9.0)  # Assume ±10mV is limit

        # 2. Check for flat/dead channel
        std_val = np.std(lead_signal)
        metrics["flat"] = bool(std_val < 0.01)  # Very low variance

        # 3. Baseline wander: high-pass filter response
        # Extract slowly varying component (baseline)
        sos = sp_signal.butter(3, 0.5, btype='highpass', fs=self.fs, output='sos')
        highpass = sp_signal.sosfilt(sos, lead_signal)
        baseline = lead_signal - highpass
        metrics["baseline_wander_mv"] = float(np.max(np.abs(baseline)))

        # 4. Noise estimation: measure high-frequency content
        # High-pass at 100 Hz to isolate noise
        if self.fs >= 200:
            try:
                sos_noise = sp_signal.butter(3, 100, btype='highpass', fs=self.fs, output='sos')
                noise_component = sp_signal.sosfilt(sos_noise, lead_signal)
                noise_power = np.mean(noise_component ** 2)
                signal_power = np.mean(highpass ** 2)
                metrics["snr_db"] = float(10 * np.log10(signal_power / (noise_power + 1e-10)))
            except Exception:
                metrics["snr_db"] = 0.0
        else:
            metrics["snr_db"] = 0.0

        # 5. Quality score (0-100)
        score = 100.0
        if metrics["saturated"]:
            score -= 50
        if metrics["flat"]:
            score -= 50
        if metrics["baseline_wander_mv"] > 1.0:
            score -= 20
        if metrics["snr_db"] < 5:
            score -= 15
        metrics["quality_score"] = max(0.0, score)

        return metrics

    def analyze_all_leads(self, signal_array: np.ndarray) -> dict:
        """
        Analyzes quality for all 12 leads.

        Args:
            signal_array: (samples, 12) array

        Returns:
            {
                "lead_quality": {lead_name: quality_score, ...},
                "overall_quality": float,  # Min quality across all leads
                "unusable_leads": [list of lead names with quality < 30],
            }
        """
        lead_names = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

        quality_by_lead = {}
        for i, name in enumerate(lead_names):
            sqi = self.compute_sqi(signal_array[:, i])
            quality_by_lead[name] = sqi["quality_score"]

        overall = min(quality_by_lead.values())
        unusable = [name for name, score in quality_by_lead.items() if score < 30]

        return {
            "lead_quality": quality_by_lead,
            "overall_quality": overall,
            "unusable_leads": unusable,
        }
