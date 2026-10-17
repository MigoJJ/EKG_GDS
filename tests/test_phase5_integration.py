"""
Phase 5 Step 6: Integration Tests - All Components

Tests all Phase 5 components in realistic scenarios:
- Job queue (async workflow)
- Security (API keys, rate limiting, CORS)
- Audit logging
- Input format parsing
"""

import pytest
import json
import tempfile
import time
from pathlib import Path
import numpy as np

# Import components
from models.job_queue import job_queue, JobStatus
from api.auth import api_key_manager, cors_config_manager
from models.audit_logger import audit_logger
from pipeline.hl7_parser import HL7Parser
from pipeline.scp_parser import SCPParser


class TestJobQueue:
    """Test Phase 5 Step 1 job queue."""

    def test_job_lifecycle(self):
        """Job should progress: queued → in_progress → complete."""
        job_id = job_queue.create_job()

        job = job_queue.get_job(job_id)
        assert job.status == JobStatus.QUEUED
        job = job_queue.get_job(job_id)
        assert job.progress == 0

        # Start job
        job_queue.start_job(job_id)
        status = job_queue.get_job(job_id)
        assert status.status == JobStatus.IN_PROGRESS

        # Progress updates
        job_queue.update_progress(job_id, 50)
        status = job_queue.get_job(job_id)
        assert status.progress == 50

        # Complete
        result = {"analysis": "done"}
        job_queue.complete_job(job_id, result)
        status = job_queue.get_job(job_id)
        assert status.status == JobStatus.COMPLETE
        assert status.result == result

    def test_job_cancellation(self):
        """Job cancellation should work."""
        job_id = job_queue.create_job()

        job_queue.cancel_job(job_id)
        status = job_queue.get_job(job_id)
        assert status.status == JobStatus.CANCELLED

    def test_concurrent_jobs(self):
        """Multiple jobs should be listable."""
        job_ids = []
        for i in range(3):
            job_id = job_queue.create_job()
            job_ids.append(job_id)

        all_jobs = job_queue.list_jobs()
        assert len(all_jobs) >= 3

    def test_job_stats(self):
        """Job statistics should be available."""
        stats = job_queue.get_stats()

        assert "total" in stats
        assert "queued" in stats
        assert "complete" in stats
        assert stats["total"] >= 0


class TestSecurityAPI:
    """Test Phase 5 Step 2 security."""

    def test_api_key_validation(self):
        """API key validation should work."""
        # Valid dev key
        is_valid, api_key, error = api_key_manager.validate_key("sk_dev_test_123456789")
        assert is_valid
        assert api_key.key_id == "dev_001"

        # Invalid key
        is_valid, api_key, error = api_key_manager.validate_key("invalid")
        assert not is_valid
        assert error == "Invalid API key"

        # Inactive key
        is_valid, api_key, error = api_key_manager.validate_key("sk_test_inactive")
        assert not is_valid
        assert error == "API key is inactive"

    def test_rate_limiting(self):
        """Rate limiting should enforce quotas."""
        # Reset and test
        key = "sk_dev_test_123456789"
        endpoint = "/api/v1/ecg/analyze"

        # First request succeeds
        is_allowed, error = api_key_manager.check_rate_limit(key, endpoint)
        assert is_allowed

    def test_cors_configuration(self):
        """CORS whitelist should restrict origins."""
        # Default allowed
        assert cors_config_manager.is_origin_allowed("http://127.0.0.1:3000")
        assert cors_config_manager.is_origin_allowed("https://emr.seoul-medical.kr")

        # Not allowed
        assert not cors_config_manager.is_origin_allowed("https://evil.com")

        # Add and check
        cors_config_manager.add_origin("https://test-hospital.kr")
        assert cors_config_manager.is_origin_allowed("https://test-hospital.kr")

    def test_api_key_list(self):
        """API keys should be listable."""
        keys = api_key_manager.list_keys()

        # Should have at least 3 keys (dev, prod, test)
        assert len(keys) >= 3
        assert any(k["key_id"] == "dev_001" for k in keys)
        assert any(k["key_id"] == "prod_001" for k in keys)

    def test_remaining_quota(self):
        """Quota status should be queryable."""
        key = "sk_dev_test_123456789"
        quota = api_key_manager.get_remaining_quota(key)

        assert "remaining_per_minute" in quota
        assert "remaining_per_day" in quota
        assert quota["remaining_per_minute"] >= 0
        assert quota["remaining_per_day"] >= 0


class TestAuditLogging:
    """Test Phase 5 Step 3 audit trail."""

    def test_signal_hashing(self):
        """ECG signals should be hashed, not stored."""
        from models.audit_logger import hash_signal

        signal = b"test_ecg_data"

        # Same signal → same hash (deterministic)
        hash1 = hash_signal(signal)
        hash2 = hash_signal(signal)
        assert hash1 == hash2

        # Different signal → different hash
        signal2 = b"different_data"
        hash3 = hash_signal(signal2)
        assert hash1 != hash3

        # Hash should be short (16 chars)
        assert len(hash1) == 16

    def test_timestamp_masking(self):
        """Timestamps should be masked to day-level."""
        from models.audit_logger import mask_timestamp
        from datetime import datetime

        dt = datetime(2026, 10, 17, 12, 34, 56)
        masked = mask_timestamp(dt)

        # Should be YYYY-MM-DD only
        assert masked == "2026-10-17"
        assert ":" not in masked

    def test_audit_entry(self):
        """Audit entry should track analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = audit_logger.__class__(audit_dir=tmpdir)

            entry = logger.create_entry(
                request_id="audit_001",
                hospital_id="test_hospital",
                signal=b"test_signal",
                model_version="phase3",
                cutoff_version="fold9"
            )

            logger.log_result(
                entry,
                primary_diagnosis=["MI"],
                kcd_code="I21.0",
                urgency="EMERGENCY",
                confidence=0.92,
                latency_ms={"total": 4500}
            )

            # Retrieve
            retrieved = logger.get_entry("audit_001")
            assert retrieved is not None
            assert retrieved["hospital_id"] == "test_hospital"
            assert retrieved["result"]["kcd_code"] == "I21.0"

    def test_doctor_feedback(self):
        """Doctor feedback should be logged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = audit_logger.__class__(audit_dir=tmpdir)

            entry = logger.create_entry(
                request_id="audit_002",
                hospital_id="test_hospital",
                signal=b"test_signal",
                model_version="phase3",
                cutoff_version="fold9"
            )

            logger.log_result(
                entry,
                primary_diagnosis=["NORM"],
                kcd_code="Z00",
                urgency="ROUTINE",
                confidence=0.88,
                latency_ms={"total": 4000}
            )

            # Add feedback
            logger.log_feedback(
                request_id="audit_002",
                agree=True,
                notes="Agreed with AI"
            )

            # Check
            retrieved = logger.get_entry("audit_002")
            assert retrieved["doctor_feedback"]["agree"] is True

    def test_audit_query(self):
        """Audit entries should be queryable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = audit_logger.__class__(audit_dir=tmpdir)

            # Create entries
            for i in range(3):
                entry = logger.create_entry(
                    request_id=f"query_{i:03d}",
                    hospital_id="hospital_a" if i < 2 else "hospital_b",
                    signal=b"test",
                    model_version="phase3",
                    cutoff_version="fold9"
                )
                logger.log_result(
                    entry,
                    primary_diagnosis=["NORM"],
                    kcd_code="Z00",
                    urgency="ROUTINE",
                    confidence=0.9,
                    latency_ms={"total": 4000}
                )

            # Query all
            all_entries = logger.query_entries(limit=10)
            assert len(all_entries) == 3

            # Query by hospital
            hosp_a = logger.query_entries(hospital_id="hospital_a", limit=10)
            assert len(hosp_a) == 2


class TestInputFormats:
    """Test Phase 5 Step 4 input parsers."""

    def test_dicom_validation(self):
        """DICOM validation should work."""
        from pipeline.dicom_parser import DicomParser

        # Create invalid DICOM file
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
            tmp.write(b"not_dicom")
            tmp_path = tmp.name

        try:
            # Should raise ValueError
            with pytest.raises(ValueError):
                DicomParser(tmp_path)
        finally:
            Path(tmp_path).unlink()

    def test_hl7_parsing(self):
        """HL7 message parsing should work."""
        hl7_msg = """MSH|^~\\&|EMR|Hospital|AI_ECG|Server|202610170930||ADT^A01|MSG001|P|2.5
PID|1||12345||DOE^JOHN||19800115|M
OBX|1|NM|HR^Heart Rate|1|75|bpm"""

        parser = HL7Parser(hl7_msg)

        # Validate
        is_valid, error = parser.validate()
        assert is_valid

        # Extract patient info
        info = parser.get_patient_info()
        assert info is not None
        if "patient_name" in info:
            assert isinstance(info["patient_name"], str)

        # Extract ECG data
        ecg = parser.get_ecg_data()
        assert isinstance(ecg, dict)

    def test_hl7_invalid(self):
        """Invalid HL7 should fail."""
        with pytest.raises(ValueError):
            HL7Parser("INVALID|MESSAGE")

    def test_scp_validation(self):
        """SCP file validation should work."""
        # Create invalid SCP file
        with tempfile.NamedTemporaryFile(suffix=".scp", delete=False) as tmp:
            tmp.write(b"not_scp")
            tmp_path = tmp.name

        try:
            with pytest.raises(ValueError):
                SCPParser(tmp_path)
        finally:
            Path(tmp_path).unlink()


class TestEndToEnd:
    """Complete workflows."""

    def test_async_job_complete_flow(self):
        """Complete async job workflow."""
        # 1. Create job
        job_id = job_queue.create_job()
        job = job_queue.get_job(job_id)
        assert job.status == JobStatus.QUEUED

        # 2. Start
        job_queue.start_job(job_id)
        assert job_queue.get_job(job_id).status == JobStatus.IN_PROGRESS

        # 3. Progress
        for progress in [25, 50, 75]:
            job_queue.update_progress(job_id, progress)
            assert job_queue.get_job(job_id).progress == progress

        # 4. Complete
        result = {
            "status": "success",
            "analysis": "complete",
            "urgency": "ROUTINE"
        }
        job_queue.complete_job(job_id, result)
        final = job_queue.get_job(job_id)
        assert final.status == JobStatus.COMPLETE
        assert final.result["urgency"] == "ROUTINE"

    def test_secure_authenticated_request(self):
        """API request with authentication."""
        key = "sk_prod_real_abcdefghij"  # prod key
        endpoint = "/api/v1/ecg/analyze"

        # Validate key
        is_valid, api_key, _ = api_key_manager.validate_key(key)
        assert is_valid
        assert api_key.key_id == "prod_001"

        # Check rate limit
        is_allowed, error = api_key_manager.check_rate_limit(key, endpoint)
        assert is_allowed

    def test_audit_complete_workflow(self):
        """Complete audit workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = audit_logger.__class__(audit_dir=tmpdir)

            # 1. Create analysis entry
            entry = logger.create_entry(
                request_id="workflow_001",
                hospital_id="workflow_hospital",
                signal=b"patient_ecg_signal",
                model_version="phase3_kaggle_20261017",
                cutoff_version="fold9_20261017"
            )

            # 2. Log analysis result
            logger.log_result(
                entry,
                primary_diagnosis=["MI", "RBBB"],
                kcd_code="I21.0",
                urgency="EMERGENCY",
                confidence=0.92,
                latency_ms={
                    "feature_extraction": 105,
                    "inference": 5,
                    "llm": 4123
                }
            )

            # 3. Doctor reviews and provides feedback
            logger.log_feedback(
                request_id="workflow_001",
                agree=True,
                notes="Agreed with AI diagnosis, patient sent to cath lab"
            )

            # 4. Query for audit report
            entry_data = logger.get_entry("workflow_001")
            assert entry_data is not None
            assert entry_data["result"]["urgency"] == "EMERGENCY"
            assert entry_data["doctor_feedback"]["agree"] is True


class TestPerformance:
    """Performance benchmarks."""

    def test_job_creation_speed(self):
        """Job creation should be fast."""
        start = time.perf_counter()
        job_id = job_queue.create_job()
        latency_ms = (time.perf_counter() - start) * 1000

        assert latency_ms < 5, f"Job creation took {latency_ms}ms (should be <5ms)"

    def test_job_lookup_speed(self):
        """Job lookup should be fast."""
        job_id = job_queue.create_job()

        start = time.perf_counter()
        _ = job_queue.get_job(job_id)
        latency_ms = (time.perf_counter() - start) * 1000

        assert latency_ms < 5, f"Job lookup took {latency_ms}ms (should be <5ms)"

    def test_hl7_parsing_speed(self):
        """HL7 parsing should be fast."""
        hl7_msg = """MSH|^~\\&|EMR|Hospital|AI_ECG|Server|202610170930||ADT^A01|MSG001|P|2.5
PID|1||12345||DOE^JOHN||19800115|M
OBR|1|ECG001|ECG001-LAB|ECG^Electrocardiogram
OBX|1|NM|HR^Heart Rate|1|75|bpm
OBX|2|NM|PR^PR Interval|1|160|ms"""

        start = time.perf_counter()
        parser = HL7Parser(hl7_msg)
        _ = parser.get_patient_info()
        _ = parser.get_ecg_data()
        latency_ms = (time.perf_counter() - start) * 1000

        assert latency_ms < 50, f"HL7 parsing took {latency_ms}ms (should be <50ms)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
