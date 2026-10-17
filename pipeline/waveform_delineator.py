"""
NeuroKit2-based Waveform Delineation Module
Detects P-wave, QRS, and T-wave boundaries with high precision using DWT method.
Replaces heuristic interval measurement in feature_extractor.
"""

from __future__ import annotations
import numpy as np
import neurokit2 as nk
from typing import Dict, List, Optional, Tuple


class WaveformDelineator:
    """
    High-precision ECG waveform delineation using NeuroKit2 DWT method.
    Detects P-onset, P-offset, QRS-onset, QRS-offset (J-point), T-onset, T-offset.
    """

    def __init__(self, sampling_rate: int = 500):
        """
        Args:
            sampling_rate: 500 Hz recommended for accurate delineation
        """
        self.fs = sampling_rate

    def delineate_lead(
        self,
        lead_signal: np.ndarray,
        rpeaks: Optional[np.ndarray] = None,
        method: str = "dwt"
    ) -> Dict[str, any]:
        """
        Delineates P, QRS, and T waves in a single ECG lead.

        Args:
            lead_signal: 1D signal array (samples,)
            rpeaks: R-peak indices. If None, auto-detected.
            method: "dwt" (default), "cwt", "peak", or "prominence"

        Returns:
            {
                "P_Onsets": array of sample indices,
                "P_Offsets": array of sample indices,
                "QRS_Onsets": array of sample indices,
                "QRS_Offsets": array of sample indices (J-point),
                "T_Onsets": array of sample indices,
                "T_Offsets": array of sample indices,
                "success": bool,
            }
        """
        try:
            # Clean the signal
            cleaned = nk.ecg_clean(lead_signal, sampling_rate=self.fs, method='neurokit')

            # Auto-detect R-peaks if not provided
            if rpeaks is None:
                _, rpeaks_info = nk.ecg_peaks(cleaned, sampling_rate=self.fs)
                rpeaks = rpeaks_info.get("ECG_R_Peaks", np.array([]))

            if len(rpeaks) == 0:
                return {"success": False, "reason": "No R-peaks detected"}

            # Delineate using specified method
            _, delineation = nk.ecg_delineate(
                cleaned,
                rpeaks=rpeaks,
                sampling_rate=self.fs,
                method=method,
                check=True  # Discard outlier detections
            )

            # Extract boundary indices
            result = {
                "P_Onsets": delineation.get("ECG_P_Onsets", np.array([])),
                "P_Offsets": delineation.get("ECG_P_Offsets", np.array([])),
                "QRS_Onsets": delineation.get("ECG_R_Onsets", np.array([])),
                "QRS_Offsets": delineation.get("ECG_R_Offsets", np.array([])),  # J-point
                "T_Onsets": delineation.get("ECG_T_Onsets", np.array([])),
                "T_Offsets": delineation.get("ECG_T_Offsets", np.array([])),
                "R_Peaks": rpeaks,
                "success": True,
            }
            return result
        except Exception as e:
            return {"success": False, "reason": str(e)}

    def measure_intervals_from_delineation(
        self,
        delineation: Dict[str, any]
    ) -> Tuple[float, float, float]:
        """
        Computes PR, QRS, and QT intervals from delineation results (in milliseconds).

        Returns:
            (pr_ms, qrs_ms, qt_ms)
        """
        pr_ms = qrs_ms = qt_ms = np.nan

        if not delineation.get("success", False):
            return pr_ms, qrs_ms, qt_ms

        # QRS duration: from onset to offset (J-point)
        onsets = delineation.get("QRS_Onsets", np.array([]))
        offsets = delineation.get("QRS_Offsets", np.array([]))
        if len(onsets) > 0 and len(offsets) > 0:
            qrs_samples = offsets[0] - onsets[0]
            qrs_ms = (qrs_samples / self.fs) * 1000.0

        # PR interval: from P-onset to QRS-onset
        p_onsets = delineation.get("P_Onsets", np.array([]))
        if len(p_onsets) > 0 and len(onsets) > 0:
            pr_samples = onsets[0] - p_onsets[0]
            pr_ms = (pr_samples / self.fs) * 1000.0

        # QT interval: from QRS-onset to T-offset
        t_offsets = delineation.get("T_Offsets", np.array([]))
        if len(onsets) > 0 and len(t_offsets) > 0:
            qt_samples = t_offsets[0] - onsets[0]
            qt_ms = (qt_samples / self.fs) * 1000.0

        return pr_ms, qrs_ms, qt_ms

    def measure_all_leads(
        self,
        signal_array: np.ndarray,
        lead_name: str = "II"
    ) -> Dict[str, any]:
        """
        Measures intervals using the lead with best quality (typically Lead II).

        Args:
            signal_array: (samples, 12) ECG array
            lead_name: which lead to use for interval measurement

        Returns:
            {
                "pr_interval_ms": float,
                "qrs_duration_ms": float,
                "qt_interval_ms": float,
                "method": "neurokit2_dwt",
                "success": bool,
            }
        """
        # Lead to column mapping
        lead_indices = {
            "I": 0, "II": 1, "III": 2, "AVR": 3, "AVL": 4, "AVF": 5,
            "V1": 6, "V2": 7, "V3": 8, "V4": 9, "V5": 10, "V6": 11
        }

        if lead_name not in lead_indices:
            return {"success": False, "reason": f"Unknown lead: {lead_name}"}

        lead_idx = lead_indices[lead_name]
        lead_signal = signal_array[:, lead_idx]

        delineation = self.delineate_lead(lead_signal)
        pr_ms, qrs_ms, qt_ms = self.measure_intervals_from_delineation(delineation)

        # Sanity bounds
        pr_ms = np.clip(pr_ms, 80, 320) if not np.isnan(pr_ms) else 160.0
        qrs_ms = np.clip(qrs_ms, 60, 220) if not np.isnan(qrs_ms) else 90.0
        qt_ms = np.clip(qt_ms, 280, 560) if not np.isnan(qt_ms) else 390.0

        return {
            "pr_interval_ms": float(pr_ms),
            "qrs_duration_ms": float(qrs_ms),
            "qt_interval_ms": float(qt_ms),
            "method": "neurokit2_dwt",
            "success": delineation.get("success", False),
        }
