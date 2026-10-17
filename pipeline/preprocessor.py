"""
ECG Signal Preprocessing Module
Implements zero-phase bandpass filtering, notch filtering, and baseline wander correction for 12-lead ECG.
"""

from __future__ import annotations
import numpy as np
from scipy import signal


class ECGPreprocessor:
    """Zero-phase digital filter and baseline correction pipeline for multi-channel ECG signals."""

    def __init__(
        self,
        sampling_rate: int = 100,
        lowcut: float = 0.5,
        highcut: float = 45.0,
        notch_freq: float = 60.0,
        filter_order: int = 3,
    ):
        self.fs = sampling_rate
        self.lowcut = lowcut
        self.highcut = highcut
        self.notch_freq = notch_freq
        self.filter_order = filter_order

        # Design Butterworth bandpass filter coefficients
        nyquist = 0.5 * self.fs
        # Ensure highcut is safely below Nyquist
        safe_highcut = min(self.highcut, nyquist - 1.0)
        low = self.lowcut / nyquist
        high = safe_highcut / nyquist
        
        self.b_band, self.a_band = signal.butter(
            self.filter_order, [low, high], btype="bandpass"
        )

        # Design IIR Notch filter (if notch_freq is below Nyquist)
        if self.notch_freq < nyquist:
            self.b_notch, self.a_notch = signal.iirnotch(
                w0=self.notch_freq, Q=30.0, fs=self.fs
            )
        else:
            self.b_notch, self.a_notch = None, None

    def filter_signal(self, raw_signal: np.ndarray) -> np.ndarray:
        """
        Applies zero-phase bandpass and notch filtering across all leads.
        Args:
            raw_signal: (samples, n_channels) numpy array in mV.
        Returns:
            filtered_signal: (samples, n_channels) filtered array preserving original mV scale.
        """
        if raw_signal.ndim == 1:
            raw_signal = raw_signal[:, np.newaxis]

        samples, channels = raw_signal.shape
        filtered = np.zeros_like(raw_signal, dtype=np.float64)

        # Minimum length required for filtfilt is filter_order * 3 + 1
        padlen = min(samples - 1, 3 * self.filter_order)

        for ch in range(channels):
            ch_data = raw_signal[:, ch].astype(np.float64)
            # 1. Bandpass filter
            try:
                ch_filtered = signal.filtfilt(
                    self.b_band, self.a_band, ch_data, padlen=padlen
                )
            except Exception:
                ch_filtered = ch_data

            # 2. Notch filter (if applicable)
            if self.b_notch is not None and self.a_notch is not None:
                try:
                    ch_filtered = signal.filtfilt(
                        self.b_notch, self.a_notch, ch_filtered, padlen=padlen
                    )
                except Exception:
                    pass

            filtered[:, ch] = ch_filtered

        return filtered

    def remove_baseline_wander(self, signal_data: np.ndarray, window_sec: float = 0.2) -> np.ndarray:
        """
        Removes remaining slow baseline drift by subtracting rolling median / polynomial trend.
        """
        window_size = int(self.fs * window_sec)
        if window_size % 2 == 0:
            window_size += 1

        if window_size >= signal_data.shape[0]:
            return signal_data

        corrected = np.zeros_like(signal_data)
        for ch in range(signal_data.shape[1]):
            # First pass: median filter to estimate QRS baseline
            trend = signal.medfilt(signal_data[:, ch], kernel_size=min(window_size, 51))
            corrected[:, ch] = signal_data[:, ch] - trend
            
        return corrected
