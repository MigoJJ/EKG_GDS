"""
Phase 5 Step 3: Audit Logging Tests

Tests for audit trail creation, storage, and retrieval.
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime
from models.audit_logger import AuditLogger, AuditEntry, hash_signal, mask_timestamp


class TestHashSignal:
    """Test ECG signal hashing (never store raw signal)."""

    def test_hash_signal_deterministic(self):
        """Same signal produces same hash."""
        signal = b"test_ecg_signal_12345"
        hash1 = hash_signal(signal)
        hash2 = hash_signal(signal)
        assert hash1 == hash2

    def test_hash_signal_different_signals(self):
        """Different signals produce different hashes."""
        signal1 = b"test_signal_1"
        signal2 = b"test_signal_2"
        assert hash_signal(signal1) != hash_signal(signal2)

    def test_hash_signal_is_short(self):
        """Hash is truncated to 16 chars for brevity."""
        signal = b"test_signal"
        h = hash_signal(signal)
        assert len(h) == 16


class TestMaskTimestamp:
    """Test timestamp masking (day granularity for privacy)."""

    def test_mask_timestamp_day_only(self):
        """Timestamp is masked to day only."""
        dt = datetime(2026, 10, 17, 12, 34, 56)
        masked = mask_timestamp(dt)
        assert masked == "2026-10-17"
        assert ":" not in masked  # No time component

    def test_mask_timestamp_default(self):
        """Default uses current time."""
        masked = mask_timestamp()
        assert len(masked) == 10
        assert masked.count("-") == 2


class TestAuditEntry:
    """Test individual audit entry lifecycle."""

    def test_create_entry(self):
        """Create audit entry."""
        entry = AuditEntry(
            request_id="req_123",
            hospital_id="seoul_med",
            signal_hash="abc123def456",
            model_version="phase3_20261017",
            cutoff_version="fold9_20261017",
        )

        assert entry.request_id == "req_123"
        assert entry.hospital_id == "seoul_med"
        assert entry.result is None
        assert entry.doctor_feedback is None

    def test_set_result(self):
        """Set analysis result."""
        entry = AuditEntry(
            request_id="req_123",
            hospital_id="seoul_med",
            signal_hash="abc123",
            model_version="phase3",
            cutoff_version="fold9",
        )

        entry.set_result(
            primary_diagnosis=["MI", "RBBB"],
            kcd_code="I21.0",
            urgency="EMERGENCY",
            confidence=0.92,
            latency_ms={"extraction": 105, "inference": 5, "llm": 4123}
        )

        assert entry.result["primary_diagnosis"] == ["MI", "RBBB"]
        assert entry.result["kcd_code"] == "I21.0"
        assert entry.result["urgency"] == "EMERGENCY"
        assert entry.result["confidence"] == 0.92

    def test_set_doctor_feedback(self):
        """Set doctor feedback."""
        entry = AuditEntry(
            request_id="req_123",
            hospital_id="seoul_med",
            signal_hash="abc123",
            model_version="phase3",
            cutoff_version="fold9",
        )

        entry.set_doctor_feedback(
            agree=True,
            notes="Agreed with AI diagnosis, immediate PCI recommended"
        )

        assert entry.doctor_feedback["agree"] is True
        assert entry.doctor_feedback["notes"] == "Agreed with AI diagnosis, immediate PCI recommended"

    def test_to_dict_excludes_signal(self):
        """Dict serialization never includes raw signal."""
        entry = AuditEntry(
            request_id="req_123",
            hospital_id="seoul_med",
            signal_hash="abc123def456",
            model_version="phase3",
            cutoff_version="fold9",
        )

        entry_dict = entry.to_dict()
        assert "signal" not in entry_dict
        assert "signal_hash" in entry_dict  # Hash is OK
        assert entry_dict["signal_hash"] == "abc123def456"

    def test_to_json(self):
        """JSON serialization is valid."""
        entry = AuditEntry(
            request_id="req_123",
            hospital_id="seoul_med",
            signal_hash="abc123",
            model_version="phase3",
            cutoff_version="fold9",
        )

        json_str = entry.to_json()
        parsed = json.loads(json_str)
        assert parsed["request_id"] == "req_123"


class TestAuditLogger:
    """Test audit logger with JSONL file storage."""

    def test_create_entry_in_logger(self):
        """Create entry via logger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(audit_dir=tmpdir)

            signal = b"test_signal_data"
            entry = logger.create_entry(
                request_id="req_001",
                hospital_id="seoul_med",
                signal=signal,
                model_version="phase3",
                cutoff_version="fold9",
            )

            assert entry.request_id == "req_001"
            assert entry.signal_hash == hash_signal(signal)

    def test_log_result_writes_to_file(self):
        """Log result writes to JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(audit_dir=tmpdir)

            entry = logger.create_entry(
                request_id="req_001",
                hospital_id="seoul_med",
                signal=b"test_signal",
                model_version="phase3",
                cutoff_version="fold9",
            )

            logger.log_result(
                entry,
                primary_diagnosis=["MI"],
                kcd_code="I21.0",
                urgency="EMERGENCY",
                confidence=0.92,
                latency_ms={"total": 4233}
            )

            # Check file exists
            files = list(Path(tmpdir).glob("*.jsonl"))
            assert len(files) == 1

            # Check content
            with open(files[0]) as f:
                content = f.read().strip()
                data = json.loads(content)
                assert data["request_id"] == "req_001"
                assert data["result"]["kcd_code"] == "I21.0"

    def test_log_feedback(self):
        """Log doctor feedback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(audit_dir=tmpdir)

            entry = logger.create_entry(
                request_id="req_001",
                hospital_id="seoul_med",
                signal=b"test_signal",
                model_version="phase3",
                cutoff_version="fold9",
            )

            logger.log_result(
                entry,
                primary_diagnosis=["MI"],
                kcd_code="I21.0",
                urgency="EMERGENCY",
                confidence=0.92,
                latency_ms={"total": 4233}
            )

            # Add feedback
            logger.log_feedback(
                request_id="req_001",
                agree=True,
                notes="Agreed with AI diagnosis"
            )

            # Retrieve and check
            retrieved = logger.get_entry("req_001")
            assert retrieved["doctor_feedback"]["agree"] is True
            assert "Agreed" in retrieved["doctor_feedback"]["notes"]

    def test_query_entries(self):
        """Query audit entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(audit_dir=tmpdir)

            # Create multiple entries
            for i in range(3):
                entry = logger.create_entry(
                    request_id=f"req_{i:03d}",
                    hospital_id="seoul_med" if i < 2 else "busan_hospital",
                    signal=b"test_signal",
                    model_version="phase3",
                    cutoff_version="fold9",
                )

                logger.log_result(
                    entry,
                    primary_diagnosis=["MI"],
                    kcd_code="I21.0",
                    urgency="EMERGENCY",
                    confidence=0.92,
                    latency_ms={"total": 4233}
                )

            # Query all
            all_entries = logger.query_entries(limit=10)
            assert len(all_entries) == 3

            # Query by hospital
            seoul_entries = logger.query_entries(hospital_id="seoul_med", limit=10)
            assert len(seoul_entries) == 2
            assert all(e["hospital_id"] == "seoul_med" for e in seoul_entries)

    def test_get_single_entry(self):
        """Get a single entry by request_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(audit_dir=tmpdir)

            entry = logger.create_entry(
                request_id="req_abc123",
                hospital_id="seoul_med",
                signal=b"test_signal",
                model_version="phase3",
                cutoff_version="fold9",
            )

            logger.log_result(
                entry,
                primary_diagnosis=["MI"],
                kcd_code="I21.0",
                urgency="EMERGENCY",
                confidence=0.92,
                latency_ms={"total": 4233}
            )

            # Retrieve
            retrieved = logger.get_entry("req_abc123")
            assert retrieved is not None
            assert retrieved["request_id"] == "req_abc123"

            # Non-existent
            none_entry = logger.get_entry("nonexistent")
            assert none_entry is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
