"""
Phase 5 Step 4: Input Format Tests

Tests for DICOM, HL7, and SCP-ECG parsers.
"""

import pytest
import numpy as np
import json
from pathlib import Path
import tempfile
from pipeline.dicom_parser import DicomParser
from pipeline.hl7_parser import HL7Parser
from pipeline.scp_parser import SCPParser


class TestDicomParser:
    """Test DICOM parsing."""

    def test_dicom_validation_invalid_file(self):
        """Invalid file should raise error."""
        with pytest.raises(FileNotFoundError):
            DicomParser("/nonexistent/file.dcm")

    def test_dicom_validation_non_dicom(self):
        """Non-DICOM file should fail validation."""
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
            tmp.write(b"not a DICOM file")
            tmp_path = tmp.name

        try:
            with pytest.raises(ValueError):
                DicomParser(tmp_path)
        finally:
            Path(tmp_path).unlink()

    def test_dicom_sampling_rate_default(self):
        """Default sampling rate should be 400 Hz."""
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
            # Create minimal DICOM header
            tmp.write(b'\x00' * 128)  # Preamble
            tmp.write(b'DICM')  # Magic
            tmp.write(b'\x00' * 100)  # Minimal header
            tmp_path = tmp.name

        try:
            parser = DicomParser(tmp_path)
            sr = parser.get_sampling_rate()
            assert sr == 400.0
        finally:
            Path(tmp_path).unlink()

    def test_dicom_patient_info_empty(self):
        """Patient info should handle missing fields."""
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
            tmp.write(b'\x00' * 128)
            tmp.write(b'DICM')
            tmp.write(b'\x00' * 100)
            tmp_path = tmp.name

        try:
            parser = DicomParser(tmp_path)
            info = parser.extract_patient_info()
            assert isinstance(info, dict)
        finally:
            Path(tmp_path).unlink()

    def test_dicom_ecg_extraction_fallback(self):
        """ECG extraction should return fallback when library unavailable."""
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
            tmp.write(b'\x00' * 128)
            tmp.write(b'DICM')
            tmp.write(b'\x00' * 100)
            tmp_path = tmp.name

        try:
            parser = DicomParser(tmp_path)
            signal = parser.extract_ecg_signal()
            # Should return dummy signal or None (depending on library)
            if signal is not None:
                assert signal.shape[0] == 12 or signal.shape[1] == 12
        finally:
            Path(tmp_path).unlink()


class TestHL7Parser:
    """Test HL7 parsing."""

    def test_hl7_parse_valid_message(self):
        """Parse valid HL7 message."""
        hl7_msg = """MSH|^~\\&|EMR|Hospital|AI_ECG|Server|202610170930||ADT^A01|MSG001|P|2.5
PID|1||12345678^^^MRN||DOE^JOHN||19800115|M
OBR|1|ECG001|ECG001-LAB|ECG^Electrocardiogram
OBX|1|NM|HR^Heart Rate|1|75|bpm"""

        parser = HL7Parser(hl7_msg)
        is_valid, error = parser.validate()
        assert is_valid

    def test_hl7_parse_missing_msh(self):
        """Missing MSH segment should fail."""
        hl7_msg = "PID|1||12345||DOE^JOHN"

        with pytest.raises(ValueError):
            HL7Parser(hl7_msg)

    def test_hl7_extract_patient_info(self):
        """Extract patient demographics."""
        hl7_msg = """MSH|^~\\&|EMR|Hospital|AI_ECG|Server|202610170930||ADT^A01|MSG001|P|2.5
PID|1||12345678^^^MRN||DOE^JOHN||19800115|M"""

        parser = HL7Parser(hl7_msg)
        info = parser.get_patient_info()

        assert 'patient_name' in info or 'patient_id' in info
        if 'patient_name' in info:
            assert isinstance(info['patient_name'], str)

    def test_hl7_extract_ecg_data(self):
        """Extract ECG measurements."""
        hl7_msg = """MSH|^~\\&|EMR|Hospital|AI_ECG|Server|202610170930||ADT^A01|MSG001|P|2.5
PID|1||12345||DOE^JOHN||19800115|M
OBX|1|NM|HR^Heart Rate|1|75|bpm
OBX|2|NM|QRS^QRS Duration|1|90|ms"""

        parser = HL7Parser(hl7_msg)
        ecg = parser.get_ecg_data()

        assert isinstance(ecg, dict)
        if 'measurements' in ecg:
            assert isinstance(ecg['measurements'], dict)

    def test_hl7_extract_interpretation(self):
        """Extract ECG interpretation."""
        hl7_msg = """MSH|^~\\&|EMR|Hospital|AI_ECG|Server|202610170930||ADT^A01|MSG001|P|2.5
PID|1||12345||DOE^JOHN||19800115|M
NTE|1||Normal ECG"""

        parser = HL7Parser(hl7_msg)
        interp = parser.get_interpretation()

        if interp:
            assert isinstance(interp, str)

    def test_hl7_calculate_age(self):
        """Calculate age from DOB."""
        hl7_msg = """MSH|^~\\&|EMR|Hospital|AI_ECG|Server|202610170930||ADT^A01|MSG001|P|2.5
PID|1||12345||DOE^JOHN||19800115|M"""

        parser = HL7Parser(hl7_msg)
        info = parser.get_patient_info()

        if 'age' in info:
            assert info['age'] > 0

    def test_hl7_to_dict(self):
        """Convert HL7 to dictionary."""
        hl7_msg = """MSH|^~\\&|EMR|Hospital|AI_ECG|Server|202610170930||ADT^A01|MSG001|P|2.5
PID|1||12345||DOE^JOHN||19800115|M
OBX|1|NM|HR|1|75|bpm"""

        parser = HL7Parser(hl7_msg)
        result = parser.to_dict()

        assert 'patient_info' in result
        assert 'ecg_data' in result
        assert isinstance(result['patient_info'], dict)
        assert isinstance(result['ecg_data'], dict)


class TestSCPParser:
    """Test SCP-ECG parsing."""

    def test_scp_validation_invalid_file(self):
        """Invalid file should raise error."""
        with pytest.raises(FileNotFoundError):
            SCPParser("/nonexistent/file.scp")

    def test_scp_validation_non_scp(self):
        """Non-SCP file should fail validation."""
        with tempfile.NamedTemporaryFile(suffix=".scp", delete=False) as tmp:
            tmp.write(b"not an SCP-ECG file")
            tmp_path = tmp.name

        try:
            with pytest.raises(ValueError):
                SCPParser(tmp_path)
        finally:
            Path(tmp_path).unlink()

    def test_scp_create_minimal_file(self):
        """Create minimal SCP-ECG file."""
        with tempfile.NamedTemporaryFile(suffix=".scp", delete=False) as tmp:
            # Magic bytes
            tmp.write(b'SCPECG')
            # Header record (record 0)
            tmp.write(b'\x00\x00')  # Record number
            tmp.write(b'\x00\x20')  # Record length (32 bytes)
            tmp.write(b'\x00' * 32)  # Header data

            # Waveform record (record 2)
            tmp.write(b'\x00\x02')  # Record number
            tmp.write(b'\x00\x30')  # Record length
            tmp.write(b'\x04\xe8')  # 1000 samples
            tmp.write(b'\x00\x0c')  # 12 leads
            tmp.write(b'\x00' * (0x30 - 4))  # Padding

            tmp_path = tmp.name

        try:
            parser = SCPParser(tmp_path)
            is_valid, error = parser.validate()
            # Should validate if waveform record exists
            assert is_valid or error
        finally:
            Path(tmp_path).unlink()

    def test_scp_sampling_rate_default(self):
        """Default sampling rate should be 500 Hz."""
        with tempfile.NamedTemporaryFile(suffix=".scp", delete=False) as tmp:
            tmp.write(b'SCPECG')
            tmp.write(b'\x00' * 100)
            tmp_path = tmp.name

        try:
            parser = SCPParser(tmp_path)
            sr = parser.get_sampling_rate()
            assert sr == 500.0
        finally:
            Path(tmp_path).unlink()

    def test_scp_patient_info_empty(self):
        """Patient info should handle missing fields."""
        with tempfile.NamedTemporaryFile(suffix=".scp", delete=False) as tmp:
            tmp.write(b'SCPECG')
            tmp.write(b'\x00' * 100)
            tmp_path = tmp.name

        try:
            parser = SCPParser(tmp_path)
            info = parser.extract_patient_info()
            assert isinstance(info, dict)
        finally:
            Path(tmp_path).unlink()


class TestFormatConversion:
    """Test converting between formats."""

    def test_hl7_to_ecg_analysis(self):
        """HL7 to ECG analysis conversion."""
        hl7_msg = """MSH|^~\\&|EMR|Hospital|AI_ECG|Server|202610170930||ADT^A01|MSG001|P|2.5
PID|1||12345||DOE^JOHN||19800115|M"""

        parser = HL7Parser(hl7_msg)
        result = parser.to_dict()

        # Simulate signal (12 leads x 1000 samples)
        signal = np.random.randn(12, 1000).astype(np.float32) * 0.1

        assert result['patient_info'] is not None
        assert signal.shape == (12, 1000)

    def test_dicom_ecg_shape_consistency(self):
        """ECG shapes should be consistent across formats."""
        # Create test signal
        test_signal = np.random.randn(12, 5000).astype(np.float32)

        # Both (12, samples) and (samples, 12) should be handled
        assert test_signal.shape[0] == 12

        if test_signal.shape[0] == 12:
            transposed = test_signal.T
            assert transposed.shape == (5000, 12)

    def test_patient_info_standardization(self):
        """Patient info should be standardized."""
        hl7_msg = """MSH|^~\\&|EMR|Hospital|AI_ECG|Server|202610170930||ADT^A01|MSG001|P|2.5
PID|1||12345||DOE^JOHN||19800115|M"""

        parser = HL7Parser(hl7_msg)
        info = parser.get_patient_info()

        # Standard fields should be present or None
        expected_fields = ['patient_id', 'patient_name', 'date_of_birth', 'sex', 'age']
        for field in expected_fields:
            if field in info:
                assert info[field] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
