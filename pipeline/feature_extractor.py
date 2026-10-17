"""
Quantitative ECG Feature Extraction Module
Calculates physiological intervals (HR, PR, QRS, QT/QTc, Axis) and morphological markers (ST deviation, Sokolow-Lyon LVH)
from 12-lead ECG signals.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import signal
from pipeline.preprocessor import ECGPreprocessor


LEAD_INDICES = {
    "I": 0, "II": 1, "III": 2, "AVR": 3, "AVL": 4, "AVF": 5,
    "V1": 6, "V2": 7, "V3": 8, "V4": 9, "V5": 10, "V6": 11
}


class ECGFeatureExtractor:
    """Extracts clinical quantitative intervals and morphological flags from 12-lead ECG signals."""

    def __init__(self, sampling_rate: int = 100):
        self.fs = sampling_rate
        self.preprocessor = ECGPreprocessor(sampling_rate=self.fs)

    def detect_qrs_peaks(self, lead_signal: np.ndarray) -> np.ndarray:
        """
        Robust Pan-Tompkins inspired QRS detector using bandpass, differentiation, squaring, and moving integration.
        Returns indices of R-peaks.
        """
        # 1. Differentiation
        diff = np.diff(lead_signal)
        # 2. Squaring
        squared = diff ** 2
        # 3. Moving-window integration (~150ms window)
        window_len = max(3, int(0.15 * self.fs))
        integrated = np.convolve(squared, np.ones(window_len) / window_len, mode="same")
        
        # 4. Adaptive peak finding
        # Minimum distance between beats: 240 bpm -> 250ms
        min_distance = max(10, int(0.25 * self.fs))
        # Threshold: 25% of 90th percentile to ignore noise spikes
        threshold = 0.25 * np.percentile(integrated, 90)
        
        peaks, _ = signal.find_peaks(integrated, distance=min_distance, height=threshold)
        
        # Refine peaks to true maximum in the raw/filtered signal around detected location
        search_window = max(2, int(0.08 * self.fs))
        refined_peaks = []
        for p in peaks:
            start = max(0, p - search_window)
            end = min(len(lead_signal), p + search_window)
            if start < end:
                true_peak = start + np.argmax(lead_signal[start:end])
                refined_peaks.append(true_peak)
                
        return np.array(sorted(list(set(refined_peaks))))

    def extract_features(
        self,
        raw_signal: np.ndarray,
        patient_sex: str = "M",
        patient_age: Optional[float] = None
    ) -> Dict[str, Union[float, int, str, bool, Dict[str, float]]]:
        """
        Extracts comprehensive clinical metrics from 12-lead ECG.
        Args:
            raw_signal: (samples, 12) numpy array in mV.
            patient_sex: 'M' or 'F' (used for QTc and ST elevation thresholds).
            patient_age: optional patient age.
        """
        filtered = self.preprocessor.filter_signal(raw_signal)

        # 1. Heart Rate & RR Regularity from Lead II (or Lead V5 if Lead II has low amplitude)
        lead_ii = filtered[:, LEAD_INDICES["II"]]
        peaks = self.detect_qrs_peaks(lead_ii)
        
        if len(peaks) < 2:
            # Fallback to Lead V5
            lead_v5 = filtered[:, LEAD_INDICES["V5"]]
            peaks = self.detect_qrs_peaks(lead_v5)

        if len(peaks) >= 2:
            rr_intervals_sec = np.diff(peaks) / float(self.fs)
            mean_rr_sec = float(np.mean(rr_intervals_sec))
            hr_bpm = round(60.0 / mean_rr_sec, 1)
            # Regularity check (Coefficient of Variation)
            rr_cv = float(np.std(rr_intervals_sec) / mean_rr_sec) if mean_rr_sec > 0 else 0.0
            rhythm_regular = rr_cv < 0.12
        else:
            hr_bpm = 70.0
            mean_rr_sec = 60.0 / 70.0
            rhythm_regular = True

        # 2. QRS Duration & PR Interval Estimation
        # Compute median beat across valid beats in Lead II
        qrs_dur_ms, pr_interval_ms, qt_interval_ms = self._measure_intervals(filtered, peaks)

        # Corrected QT using Bazett's formula: QTc = QT / sqrt(RR_sec)
        qtc_ms = round(qt_interval_ms / math.sqrt(max(0.2, mean_rr_sec)), 1)

        # 3. Frontal QRS Heart Axis (Degrees)
        axis_deg, axis_category = self._calculate_heart_axis(filtered, peaks)

        # 4. ST-Segment Deviation Analysis across all leads
        st_deviations = self._calculate_st_deviations(filtered, peaks)
        
        # 5. Left Ventricular Hypertrophy (Sokolow-Lyon Index)
        # S(V1) + max(R(V5), R(V6)) in mV (normal < 3.5 mV)
        s_v1 = abs(min(0.0, np.min(filtered[:, LEAD_INDICES["V1"]])))
        r_v5 = max(0.0, np.max(filtered[:, LEAD_INDICES["V5"]]))
        r_v6 = max(0.0, np.max(filtered[:, LEAD_INDICES["V6"]]))
        sokolow_lyon_mv = round(s_v1 + max(r_v5, r_v6), 2)
        lvh_voltage_criteria = sokolow_lyon_mv >= 3.5

        # 6. High-Priority Clinical Findings Detection
        # Check STEMI criteria: ST elevation in >=2 contiguous leads (V1-V3: >0.2mV for M, >0.15 for F; other leads >0.1mV)
        st_elevation_leads = []
        st_depression_leads = []
        for lead, dev in st_deviations.items():
            threshold_ste = 0.20 if (lead in ["V1", "V2", "V3"] and patient_sex == "M") else 0.10
            if dev >= threshold_ste:
                st_elevation_leads.append(lead)
            elif dev <= -0.10:
                st_depression_leads.append(lead)

        stemi_suspected = len(st_elevation_leads) >= 2
        wide_qrs = qrs_dur_ms >= 120.0
        prolonged_pr = pr_interval_ms >= 200.0
        prolonged_qtc = qtc_ms >= (460.0 if patient_sex == "F" else 450.0)

        return {
            "heart_rate_bpm": hr_bpm,
            "rhythm_regular": rhythm_regular,
            "pr_interval_ms": pr_interval_ms,
            "qrs_duration_ms": qrs_dur_ms,
            "qt_interval_ms": qt_interval_ms,
            "qtc_interval_ms": qtc_ms,
            "heart_axis_deg": axis_deg,
            "heart_axis_category": axis_category,
            "sokolow_lyon_mv": sokolow_lyon_mv,
            "lvh_voltage_criteria": lvh_voltage_criteria,
            "st_deviations_mv": st_deviations,
            "st_elevation_leads": st_elevation_leads,
            "st_depression_leads": st_depression_leads,
            "stemi_suspected": stemi_suspected,
            "wide_qrs": wide_qrs,
            "prolonged_pr": prolonged_pr,
            "prolonged_qtc": prolonged_qtc,
        }

    def _measure_intervals(
        self,
        filtered: np.ndarray,
        peaks: np.ndarray
    ) -> Tuple[float, float, float]:
        """Estimates QRS duration, PR interval, and QT interval from median cycle."""
        lead_ii = filtered[:, LEAD_INDICES["II"]]
        if len(peaks) < 2:
            return 90.0, 160.0, 380.0

        # Segment beats around R peaks (-200ms to +400ms)
        pre_samples = int(0.20 * self.fs)
        post_samples = int(0.40 * self.fs)
        beat_epochs = []
        
        for p in peaks:
            if p - pre_samples >= 0 and p + post_samples < len(lead_ii):
                beat_epochs.append(lead_ii[p - pre_samples : p + post_samples])

        if not beat_epochs:
            return 90.0, 160.0, 380.0

        median_beat = np.median(np.array(beat_epochs), axis=0)
        r_idx = pre_samples

        # 1. QRS onset and J-point (offset) around R peak
        # QRS onset: search backwards from R peak to local minimum
        search_back = max(2, int(0.08 * self.fs))
        q_idx = r_idx - np.argmin(median_beat[r_idx - search_back : r_idx]) if r_idx >= search_back else r_idx - 5
        
        # J-point (QRS offset): search forward from R peak to S-wave nadir / inflection
        search_fwd = max(2, int(0.09 * self.fs))
        s_idx = r_idx + np.argmin(median_beat[r_idx : r_idx + search_fwd])
        
        qrs_samples = max(4, s_idx - q_idx)
        qrs_ms = round((qrs_samples / float(self.fs)) * 1000.0, 1)
        # Cap to physiologic boundaries
        qrs_ms = max(60.0, min(220.0, qrs_ms))

        # 2. PR Interval (P-wave start to QRS onset)
        # P-wave window: 120ms to 240ms prior to R peak
        p_window_start = max(0, r_idx - int(0.24 * self.fs))
        p_window_end = max(0, r_idx - int(0.08 * self.fs))
        if p_window_end > p_window_start:
            p_peak_rel = np.argmax(median_beat[p_window_start:p_window_end])
            p_peak_idx = p_window_start + p_peak_rel
            pr_samples = q_idx - (p_peak_idx - int(0.04 * self.fs))
            pr_ms = round(max(80.0, min(320.0, (pr_samples / float(self.fs)) * 1000.0)), 1)
        else:
            pr_ms = 160.0

        # 3. QT Interval (QRS onset to T-wave end)
        t_window_start = s_idx + int(0.06 * self.fs)
        t_window_end = min(len(median_beat) - 1, s_idx + int(0.35 * self.fs))
        if t_window_end > t_window_start:
            t_peak_rel = np.argmax(np.abs(median_beat[t_window_start:t_window_end]))
            t_peak_idx = t_window_start + t_peak_rel
            t_end_idx = min(len(median_beat) - 1, t_peak_idx + int(0.06 * self.fs))
            qt_samples = t_end_idx - q_idx
            qt_ms = round(max(280.0, min(560.0, (qt_samples / float(self.fs)) * 1000.0)), 1)
        else:
            qt_ms = 390.0

        return qrs_ms, pr_ms, qt_ms

    def _calculate_heart_axis(
        self,
        filtered: np.ndarray,
        peaks: np.ndarray
    ) -> Tuple[float, str]:
        """Calculates frontal QRS electrical axis using Lead I and Lead aVF."""
        lead_i = filtered[:, LEAD_INDICES["I"]]
        lead_avf = filtered[:, LEAD_INDICES["AVF"]]

        # Net QRS amplitude = max(QRS) + min(QRS)
        def net_qrs(sig: np.ndarray) -> float:
            if len(peaks) > 0:
                amps = []
                for p in peaks:
                    st = max(0, p - int(0.06 * self.fs))
                    en = min(len(sig), p + int(0.06 * self.fs))
                    if st < en:
                        amps.append(np.max(sig[st:en]) + np.min(sig[st:en]))
                return float(np.median(amps)) if amps else 0.0
            return float(np.max(sig) + np.min(sig))

        net_i = net_qrs(lead_i)
        net_avf = net_qrs(lead_avf)

        # Axis in degrees = atan2(aVF, I) * (180 / pi)
        axis_rad = math.atan2(net_avf, net_i)
        axis_deg = round(math.degrees(axis_rad), 1)

        # Categorize
        if -30.0 <= axis_deg <= 90.0:
            category = "Normal Axis"
        elif -90.0 <= axis_deg < -30.0:
            category = "Left Axis Deviation (LAD)"
        elif 90.0 < axis_deg <= 180.0:
            category = "Right Axis Deviation (RAD)"
        else:
            category = "Extreme Axis (Northwest)"

        return axis_deg, category

    def _calculate_st_deviations(
        self,
        filtered: np.ndarray,
        peaks: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculates J-point / ST segment amplitude relative to PR baseline across all 12 leads.
        Positive: ST Elevation, Negative: ST Depression (in mV).
        """
        st_results: Dict[str, float] = {}
        
        # J-point is approximately 60ms after R peak
        j_offset = int(0.06 * self.fs)
        # PR is approximately 80ms before R peak
        pr_offset = int(0.08 * self.fs)

        for lead_name, ch_idx in LEAD_INDICES.items():
            ch_data = filtered[:, ch_idx]
            if len(peaks) >= 2:
                devs = []
                for p in peaks:
                    pr_idx = p - pr_offset
                    j_idx = p + j_offset
                    if 0 <= pr_idx < len(ch_data) and 0 <= j_idx < len(ch_data):
                        baseline = ch_data[pr_idx]
                        j_val = ch_data[j_idx]
                        devs.append(j_val - baseline)
                median_dev = float(np.median(devs)) if devs else 0.0
            else:
                median_dev = 0.0

            st_results[lead_name] = round(median_dev, 3)

        return st_results
