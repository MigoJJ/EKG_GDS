"""
Phase 5: Asynchronous API Tests

Tests for job queue, background tasks, and async endpoints.
"""

import pytest
import time
import numpy as np
from models.job_queue import JobQueue, Job, JobStatus
from pipeline.task_executor import TaskExecutor


class TestJobQueue:
    """Test in-memory job queue."""

    def test_create_job(self):
        """Job creation with generated ID."""
        queue = JobQueue()
        job_id = queue.create_job()
        assert job_id is not None
        assert len(job_id) > 0

        job = queue.get_job(job_id)
        assert job is not None
        assert job.status == JobStatus.QUEUED

    def test_create_job_custom_id(self):
        """Job creation with custom ID."""
        queue = JobQueue()
        job_id = queue.create_job("custom_job_001")
        assert job_id == "custom_job_001"

    def test_create_duplicate_job_fails(self):
        """Cannot create job with existing ID."""
        queue = JobQueue()
        queue.create_job("job_001")
        with pytest.raises(ValueError):
            queue.create_job("job_001")

    def test_job_lifecycle(self):
        """Job transitions: queued → in_progress → complete."""
        queue = JobQueue()
        job_id = queue.create_job()

        # Start job
        assert queue.start_job(job_id)
        job = queue.get_job(job_id)
        assert job.status == JobStatus.IN_PROGRESS
        assert job.started_at is not None

        # Update progress
        assert queue.update_progress(job_id, 50)
        job = queue.get_job(job_id)
        assert job.progress == 50

        # Complete job
        result = {"diagnosis": ["MI"], "urgency": "EMERGENCY"}
        assert queue.complete_job(job_id, result)
        job = queue.get_job(job_id)
        assert job.status == JobStatus.COMPLETE
        assert job.progress == 100
        assert job.result == result

    def test_job_failure(self):
        """Job failure with error message."""
        queue = JobQueue()
        job_id = queue.create_job()
        queue.start_job(job_id)

        error_msg = "LLM timeout"
        assert queue.fail_job(job_id, error_msg)
        job = queue.get_job(job_id)
        assert job.status == JobStatus.FAILED
        assert job.error == error_msg

    def test_job_cancellation(self):
        """Cancel queued or in-progress job."""
        queue = JobQueue()
        job_id = queue.create_job()

        # Can cancel queued job
        assert queue.cancel_job(job_id)
        job = queue.get_job(job_id)
        assert job.status == JobStatus.CANCELLED

        # Cannot cancel completed job
        job_id2 = queue.create_job()
        queue.complete_job(job_id2, {})
        assert not queue.cancel_job(job_id2)

    def test_list_jobs_unfiltered(self):
        """List all jobs."""
        queue = JobQueue()
        queue.create_job("job1")
        queue.create_job("job2")
        queue.create_job("job3")

        jobs = queue.list_jobs()
        assert len(jobs) == 3

    def test_list_jobs_filtered_by_status(self):
        """Filter jobs by status."""
        queue = JobQueue()
        job_id1 = queue.create_job()
        job_id2 = queue.create_job()

        queue.start_job(job_id1)
        queue.complete_job(job_id1, {})

        completed_jobs = queue.list_jobs(status=JobStatus.COMPLETE)
        assert len(completed_jobs) == 1
        assert completed_jobs[0]["job_id"] == job_id1

        queued_jobs = queue.list_jobs(status=JobStatus.QUEUED)
        assert len(queued_jobs) == 1
        assert queued_jobs[0]["job_id"] == job_id2

    def test_get_stats(self):
        """Queue statistics."""
        queue = JobQueue()
        job_id1 = queue.create_job()
        job_id2 = queue.create_job()
        job_id3 = queue.create_job()

        queue.start_job(job_id1)
        queue.complete_job(job_id1, {})
        queue.fail_job(job_id2, "error")

        stats = queue.get_stats()
        assert stats["total"] == 3
        assert stats["complete"] == 1
        assert stats["failed"] == 1
        assert stats["queued"] == 1
        assert stats["in_progress"] == 0

    def test_cleanup_old_jobs(self):
        """Remove old completed jobs."""
        queue = JobQueue(cleanup_interval_seconds=1)
        job_id = queue.create_job()
        queue.start_job(job_id)
        queue.complete_job(job_id, {})

        # Job just completed, should not be cleaned
        cleaned = queue.cleanup_old_jobs()
        assert cleaned == 0

        # Wait for cleanup interval
        time.sleep(1.1)
        cleaned = queue.cleanup_old_jobs()
        assert cleaned == 1
        assert queue.get_job(job_id) is None

    def test_job_to_dict(self):
        """Serialize job to dict."""
        queue = JobQueue()
        job_id = queue.create_job()
        queue.start_job(job_id)
        queue.update_progress(job_id, 75)

        job_dict = queue.get_job(job_id).to_dict()
        assert job_dict["job_id"] == job_id
        assert job_dict["status"] == "in_progress"
        assert job_dict["progress"] == 75
        assert "created_at" in job_dict
        assert "started_at" in job_dict


class TestTaskExecutor:
    """Test background task execution."""

    def test_submit_and_complete_task(self):
        """Submit task and wait for completion."""
        results = []

        def mock_task(job_id, value):
            time.sleep(0.1)
            results.append(value)

        executor = TaskExecutor(max_workers=2)
        executor.submit_task("job1", mock_task, value=42)

        # Wait for task
        assert executor.wait_for_task("job1", timeout=1.0)
        assert results == [42]

    def test_submit_multiple_tasks(self):
        """Submit and execute multiple tasks concurrently."""
        results = []
        lock = __import__("threading").Lock()

        def mock_task(job_id, delay, value):
            time.sleep(delay)
            with lock:
                results.append(value)

        executor = TaskExecutor(max_workers=3)
        executor.submit_task("job1", mock_task, delay=0.1, value=1)
        executor.submit_task("job2", mock_task, delay=0.05, value=2)
        executor.submit_task("job3", mock_task, delay=0.15, value=3)

        # Wait for all
        assert executor.wait_for_task("job1", timeout=1.0)
        assert executor.wait_for_task("job2", timeout=1.0)
        assert executor.wait_for_task("job3", timeout=1.0)

        # All completed
        assert len(results) == 3
        assert set(results) == {1, 2, 3}

    def test_task_timeout(self):
        """Task doesn't complete within timeout."""
        def slow_task(job_id):
            time.sleep(2.0)

        executor = TaskExecutor(max_workers=1)
        executor.submit_task("job1", slow_task)

        # Wait with short timeout
        result = executor.wait_for_task("job1", timeout=0.5)
        assert not result  # Timeout occurred

    def test_task_exception_handling(self):
        """Task with exception doesn't crash executor."""
        def failing_task(job_id):
            raise ValueError("Test error")

        executor = TaskExecutor(max_workers=1)
        executor.submit_task("job1", failing_task)

        # Task completes (despite exception)
        assert executor.wait_for_task("job1", timeout=1.0)

    def test_cancel_task(self):
        """Request task cancellation."""
        executor = TaskExecutor(max_workers=1)
        executor.submit_task("job1", lambda job_id: time.sleep(1.0))

        # Cancel request (doesn't forcefully stop, just signals)
        assert executor.cancel_task("job1")


class TestAsyncAPIIntegration:
    """Integration tests with FastAPI async endpoints (requires test client)."""

    @pytest.mark.asyncio
    async def test_async_endpoint_returns_job_id(self):
        """POST /analyze-async returns job_id immediately."""
        # This requires FastAPI TestClient
        # Placeholder for actual integration test
        pass

    @pytest.mark.asyncio
    async def test_poll_job_status(self):
        """GET /jobs/{job_id} returns progress."""
        # Placeholder for actual integration test
        pass

    @pytest.mark.asyncio
    async def test_cancel_job_endpoint(self):
        """DELETE /jobs/{job_id} cancels job."""
        # Placeholder for actual integration test
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
