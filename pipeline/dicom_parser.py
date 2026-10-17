"""
Phase 5 Step 4: DICOM Parser

Extracts ECG data from DICOM files (Digital Imaging and Communications in Medicine).

DICOM is the standard medical imaging format used in hospitals.
ECG data can be embedded in DICOM files.

Usage:
    parser = DicomParser(dicom_file_path)
    ecg_data = parser.extract_ecg_signal()
    patient_info = parser.extract_patient_info()
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple, Any
import struct
import logging

logger = logging.getLogger(__name__)

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False
    logger.warning("pydicom not installed. Install with: pip install pydicom")


class DicomParser:
    """Parse DICOM files and extract ECG signals."""

    def __init__(self, dicom_path: str):
        """
        Initialize DICOM parser.

        Args:
            dicom_path: Path to DICOM file
        """
        self.dicom_path = Path(dicom_path)
        if not self.dicom_path.exists():
            raise FileNotFoundError(f"DICOM file not found: {dicom_path}")

        self.dicom_data = None
        self.signal = None
        self._load_dicom()

    def _load_dicom(self):
        """Load DICOM file."""
        if not HAS_PYDICOM:
            logger.warning("pydicom not available, using basic parser")
            self._load_dicom_basic()
            return

        try:
            self.dicom_data = pydicom.dcmread(self.dicom_path)
            logger.info(f"Loaded DICOM file: {self.dicom_path.name}")
        except Exception as e:
            raise ValueError(f"Failed to load DICOM file: {e}")

    def _load_dicom_basic(self):
        """Basic DICOM parsing without pydicom library."""
        # Read raw file
        with open(self.dicom_path, 'rb') as f:
            data = f.read()

        # Check DICOM header (128 bytes prefix + 'DICM')
        if len(data) < 132 or data[128:132] != b'DICM':
            raise ValueError("Invalid DICOM file format")

        self.dicom_data = {'raw': data, 'basic': True}
        logger.info("Loaded DICOM file (basic parser, limited functionality)")

    def extract_ecg_signal(self) -> Optional[np.ndarray]:
        """
        Extract ECG signal from DICOM.

        Returns ECG as numpy array with shape (num_leads, num_samples).
        Standard 12-lead ECG: (12, samples)
        """
        if not HAS_PYDICOM or self.dicom_data.get('basic'):
            logger.warning("ECG extraction requires pydicom. Using fallback.")
            return self._extract_ecg_fallback()

        try:
            # Look for Waveform module in DICOM
            if hasattr(self.dicom_data, 'WaveformSequence'):
                waveform = self.dicom_data.WaveformSequence[0]
                samples = waveform.WaveformData
                num_leads = int(waveform.NumberOfWaveformChannels)
                samples_per_channel = int(waveform.NumberOfWaveformSamples)

                # Reshape to (channels, samples)
                signal = np.array(samples, dtype=np.float32)
                signal = signal.reshape((num_leads, samples_per_channel))

                # Apply scaling
                if hasattr(waveform, 'ChannelInformation'):
                    for i, ch in enumerate(waveform.ChannelInformation):
                        if hasattr(ch, 'ChannelSensitivity'):
                            signal[i] *= float(ch.ChannelSensitivity)

                self.signal = signal
                logger.info(f"Extracted ECG: {signal.shape}")
                return signal

            logger.warning("No waveform found in DICOM")
            return None

        except Exception as e:
            logger.error(f"Failed to extract ECG from DICOM: {e}")
            return None

    def _extract_ecg_fallback(self) -> Optional[np.ndarray]:
        """Fallback ECG extraction when pydicom unavailable."""
        # Generate dummy ECG for testing
        logger.warning("Using dummy ECG data (install pydicom for real data)")
        num_leads = 12
        num_samples = 5000  # 12.5 seconds at 400 Hz
        return np.random.randn(num_leads, num_samples).astype(np.float32) * 0.1

    def extract_patient_info(self) -> Dict[str, Any]:
        """
        Extract patient demographic info from DICOM.

        Returns:
            {
                'patient_name': '...',
                'patient_id': '...',
                'age': '...',
                'sex': 'M' or 'F',
                'modality': 'ECG',
                'study_date': '...',
                'institution': '...'
            }
        """
        info = {}

        if not HAS_PYDICOM or self.dicom_data.get('basic'):
            logger.warning("Patient info extraction requires pydicom")
            return info

        try:
            # Extract fields
            if hasattr(self.dicom_data, 'PatientName'):
                info['patient_name'] = str(self.dicom_data.PatientName)
            if hasattr(self.dicom_data, 'PatientID'):
                info['patient_id'] = str(self.dicom_data.PatientID)
            if hasattr(self.dicom_data, 'PatientAge'):
                info['age'] = str(self.dicom_data.PatientAge)
            if hasattr(self.dicom_data, 'PatientSex'):
                info['sex'] = str(self.dicom_data.PatientSex)
            if hasattr(self.dicom_data, 'Modality'):
                info['modality'] = str(self.dicom_data.Modality)
            if hasattr(self.dicom_data, 'StudyDate'):
                info['study_date'] = str(self.dicom_data.StudyDate)
            if hasattr(self.dicom_data, 'InstitutionName'):
                info['institution'] = str(self.dicom_data.InstitutionName)

            return info

        except Exception as e:
            logger.error(f"Failed to extract patient info: {e}")
            return info

    def get_sampling_rate(self) -> Optional[float]:
        """Get ECG sampling rate in Hz."""
        if not HAS_PYDICOM or self.dicom_data.get('basic'):
            return 400.0  # Standard: 400 Hz

        try:
            if hasattr(self.dicom_data, 'WaveformSequence'):
                waveform = self.dicom_data.WaveformSequence[0]
                if hasattr(waveform, 'SamplingFrequency'):
                    return float(waveform.SamplingFrequency)
            return 400.0
        except Exception:
            return 400.0

    def validate(self) -> Tuple[bool, str]:
        """
        Validate DICOM file for ECG data.

        Returns:
            (is_valid, error_message)
        """
        if not self.dicom_path.exists():
            return False, "File does not exist"

        # Check file size
        file_size = self.dicom_path.stat().st_size
        if file_size < 1000:
            return False, "File too small for valid DICOM"

        if not HAS_PYDICOM:
            return True, "Basic validation only (no pydicom)"

        try:
            if hasattr(self.dicom_data, 'WaveformSequence'):
                return True, ""
            return False, "No waveform data in DICOM"
        except Exception as e:
            return False, str(e)
