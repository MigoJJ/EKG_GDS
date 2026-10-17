"""
Phase 5 Step 4: SCP-ECG Parser

Parses SCP-ECG files (Standard Communications Protocol for ECG).

SCP-ECG is a standard format for ECG data exchange from medical devices.
It's a binary format with structured records for waveform data and metadata.

Format Overview:
    - Header record (record 0)
    - QRS locations (record 1)
    - Waveform data (record 2)
    - Patient demographics (record 3)
    - Interpretations (record 4)

Usage:
    parser = SCPParser(scp_file_path)
    signal = parser.extract_signal()
    patient_info = parser.extract_patient_info()
"""

import struct
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Any
import logging

logger = logging.getLogger(__name__)


class SCPParser:
    """Parse SCP-ECG files and extract ECG signals."""

    MAGIC = b'SCPECG'  # File magic bytes

    # Record types
    HEADER_RECORD = 0
    QRS_RECORD = 1
    WAVEFORM_RECORD = 2
    PATIENT_RECORD = 3
    INTERPRETATION_RECORD = 4
    LEAD_MEASUREMENT_RECORD = 5

    def __init__(self, scp_path: str):
        """
        Initialize SCP-ECG parser.

        Args:
            scp_path: Path to SCP-ECG file
        """
        self.scp_path = Path(scp_path)
        if not self.scp_path.exists():
            raise FileNotFoundError(f"SCP-ECG file not found: {scp_path}")

        self.records = {}
        self.signal = None
        self._parse_file()

    def _parse_file(self):
        """Parse SCP-ECG file structure."""
        with open(self.scp_path, 'rb') as f:
            data = f.read()

        # Check magic bytes
        if len(data) < 6 or data[:6] != self.MAGIC:
            raise ValueError("Invalid SCP-ECG file (missing magic bytes)")

        # Parse records
        offset = 0
        while offset < len(data):
            # Read record header (4 bytes: record number + length)
            if offset + 4 > len(data):
                break

            record_num, record_len = struct.unpack('>HH', data[offset:offset+4])
            offset += 4

            # Read record data
            if offset + record_len > len(data):
                logger.warning(f"Incomplete record {record_num}")
                break

            record_data = data[offset:offset+record_len]
            offset += record_len

            # Store record
            if record_num not in self.records:
                self.records[record_num] = []
            self.records[record_num].append(record_data)

        logger.info(f"Parsed SCP-ECG file: {len(self.records)} record types")

    def extract_signal(self) -> Optional[np.ndarray]:
        """
        Extract ECG waveform from record 2.

        Returns:
            ECG signal as numpy array with shape (num_leads, num_samples)
        """
        if self.WAVEFORM_RECORD not in self.records:
            logger.warning("No waveform record found")
            return None

        try:
            waveform_data = self.records[self.WAVEFORM_RECORD][0]

            # Parse waveform record structure
            # Bytes 0-1: Number of samples per lead
            # Bytes 2-3: Number of leads
            # Bytes 4+: Encoded waveform data

            if len(waveform_data) < 4:
                return None

            num_samples = struct.unpack('>H', waveform_data[0:2])[0]
            num_leads = struct.unpack('>H', waveform_data[2:4])[0]

            logger.info(f"Waveform: {num_leads} leads, {num_samples} samples each")

            # Parse lead information (assuming 12-lead ECG standard)
            # Each sample is 2 bytes (int16)
            samples_bytes = 4 + (num_leads * 2 * num_samples)  # 2 bytes per sample

            if len(waveform_data) < samples_bytes:
                logger.warning("Incomplete waveform data")
                return None

            # Extract samples
            signal = np.zeros((num_leads, num_samples), dtype=np.float32)

            for lead in range(num_leads):
                for sample in range(num_samples):
                    offset = 4 + (lead * num_samples + sample) * 2
                    value = struct.unpack('>h', waveform_data[offset:offset+2])[0]
                    signal[lead, sample] = float(value)

            # Apply gain/offset if available in header record
            signal = self._apply_calibration(signal)

            self.signal = signal
            return signal

        except Exception as e:
            logger.error(f"Failed to extract signal: {e}")
            return None

    def _apply_calibration(self, signal: np.ndarray) -> np.ndarray:
        """Apply calibration from header record."""
        if self.HEADER_RECORD not in self.records:
            return signal

        try:
            header = self.records[self.HEADER_RECORD][0]

            # Standard ECG calibration: 1 mV = 200 samples at 500 Hz
            # Adjust based on sample rate and encoding
            mv_per_unit = 1.0 / 200.0  # Default: 5 µV per unit

            # Look for calibration info in header (typically at offset 8)
            if len(header) > 10:
                # Bytes 8-9: Calibration factor (if present)
                try:
                    cal_factor = struct.unpack('>H', header[8:10])[0]
                    if cal_factor > 0:
                        mv_per_unit = 1.0 / float(cal_factor)
                except Exception:
                    pass

            signal = signal * mv_per_unit
            return signal

        except Exception as e:
            logger.warning(f"Failed to apply calibration: {e}")
            return signal

    def extract_patient_info(self) -> Dict[str, Any]:
        """
        Extract patient demographics from record 3.

        Returns:
            Patient info dictionary
        """
        info = {}

        if self.PATIENT_RECORD not in self.records:
            return info

        try:
            patient_data = self.records[self.PATIENT_RECORD][0]

            # Parse patient record structure
            # This is a variable-length format
            # Common fields:
            # Offset 0-1: Record length
            # Offset 2+: Tag-length-value pairs

            offset = 2
            while offset < len(patient_data):
                if offset + 2 > len(patient_data):
                    break

                tag = patient_data[offset]
                field_len = patient_data[offset + 1]
                offset += 2

                if offset + field_len > len(patient_data):
                    break

                field_data = patient_data[offset:offset+field_len]
                offset += field_len

                # Decode field based on tag
                try:
                    if tag == 0:  # Patient name
                        info['patient_name'] = field_data.decode('ascii', errors='ignore')
                    elif tag == 1:  # Patient ID
                        info['patient_id'] = field_data.decode('ascii', errors='ignore')
                    elif tag == 2:  # Birthdate (YYYYMMDD)
                        dob_str = field_data.decode('ascii', errors='ignore')
                        info['date_of_birth'] = self._format_date(dob_str)
                    elif tag == 3:  # Sex (M/F/U)
                        info['sex'] = field_data.decode('ascii', errors='ignore')
                    elif tag == 4:  # Race
                        info['race'] = field_data.decode('ascii', errors='ignore')
                    elif tag == 5:  # Height
                        info['height_cm'] = int.from_bytes(field_data[:2], 'big')
                    elif tag == 6:  # Weight
                        info['weight_kg'] = int.from_bytes(field_data[:2], 'big')
                except Exception as e:
                    logger.debug(f"Error parsing patient field {tag}: {e}")

            return info

        except Exception as e:
            logger.warning(f"Failed to extract patient info: {e}")
            return info

    def extract_interpretation(self) -> Optional[str]:
        """
        Extract diagnostic interpretation from record 4.

        Returns:
            Interpretation text or None
        """
        if self.INTERPRETATION_RECORD not in self.records:
            return None

        try:
            interp_data = self.records[self.INTERPRETATION_RECORD][0]
            # Try to decode as text
            interpretation = interp_data.decode('utf-8', errors='ignore').strip()
            return interpretation if interpretation else None
        except Exception as e:
            logger.warning(f"Failed to extract interpretation: {e}")
            return None

    def get_sampling_rate(self) -> float:
        """Get sampling rate from header record."""
        if self.HEADER_RECORD not in self.records:
            return 500.0  # Default: 500 Hz

        try:
            header = self.records[self.HEADER_RECORD][0]

            # Sampling rate typically at offset 4-6
            if len(header) > 6:
                sr = struct.unpack('>H', header[4:6])[0]
                if 50 <= sr <= 10000:
                    return float(sr)

            return 500.0

        except Exception:
            return 500.0

    def _format_date(self, date_str: str) -> str:
        """Format YYYYMMDD to YYYY-MM-DD."""
        try:
            if len(date_str) >= 8:
                return f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}"
            return date_str
        except Exception:
            return date_str

    def validate(self) -> Tuple[bool, str]:
        """
        Validate SCP-ECG file.

        Returns:
            (is_valid, error_message)
        """
        if not self.scp_path.exists():
            return False, "File does not exist"

        # Check file size
        file_size = self.scp_path.stat().st_size
        if file_size < 100:
            return False, "File too small for valid SCP-ECG"

        # Check magic bytes
        try:
            with open(self.scp_path, 'rb') as f:
                magic = f.read(6)
            if magic != self.MAGIC:
                return False, "Invalid magic bytes"
        except Exception as e:
            return False, str(e)

        # Check for waveform record
        if self.WAVEFORM_RECORD in self.records:
            return True, ""

        return False, "No waveform record found"
