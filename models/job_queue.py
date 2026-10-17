"""
Phase 5: Asynchronous Job Queue for Long-Running ECG Analysis

Manages background tasks (LLM report generation) with real-time progress tracking.
In-memory queue for simplicity; upgradable to Redis for distributed deployment.

Status lifecycle:
  queued → in_progress (0%) → in_progress (25%, 50%, 75%) → complete
         → cancelled (if user interrupts)
         → failed (if error occurs)
"""

from __future__ import annotations
import uuid
import time
from typing import Any, Dict, Optional, Callable
from enum import Enum
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
import threading


class JobStatus(str, Enum):
    """Job lifecycle states."""
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """Represents a single analysis job."""
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    progress: int = 0  # 0–100%
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    result: Optional[Dict[str, Any]] = None  # Final analysis result
    error: Optional[str] = None  # Error message if failed

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        data = asdict(self)
        data["status"] = self.status.value
        data["created_at"] = self.created_at.isoformat()
        data["started_at"] = self.started_at.isoformat() if self.started_at else None
        data["completed_at"] = self.completed_at.isoformat() if self.completed_at else None
        return data


class JobQueue:
    """In-memory job queue for asynchronous ECG analysis."""

    def __init__(self, cleanup_interval_seconds: int = 3600):
        """
        Initialize job queue.

        Args:
            cleanup_interval_seconds: Auto-remove completed jobs after this duration
        """
        self.jobs: Dict[str, Job] = {}
        self.lock = threading.RLock()
        self.cleanup_interval = cleanup_interval_seconds

    def create_job(self, job_id: Optional[str] = None) -> str:
        """
        Create a new queued job.

        Args:
            job_id: Optional custom ID; defaults to UUID

        Returns:
            job_id
        """
        job_id = job_id or str(uuid.uuid4())[:8]

        with self.lock:
            if job_id in self.jobs:
                raise ValueError(f"Job {job_id} already exists")

            self.jobs[job_id] = Job(job_id=job_id, status=JobStatus.QUEUED)
            return job_id

    def get_job(self, job_id: str) -> Optional[Job]:
        """Retrieve job by ID."""
        with self.lock:
            return self.jobs.get(job_id)

    def start_job(self, job_id: str) -> bool:
        """Transition job to in_progress. Returns True if successful."""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return False
            if job.status != JobStatus.QUEUED:
                return False

            job.status = JobStatus.IN_PROGRESS
            job.started_at = datetime.utcnow()
            return True

    def update_progress(self, job_id: str, progress: int) -> bool:
        """Update job progress (0–100). Returns True if successful."""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return False
            if job.status != JobStatus.IN_PROGRESS:
                return False

            job.progress = min(100, max(0, progress))
            return True

    def complete_job(self, job_id: str, result: Dict[str, Any]) -> bool:
        """Mark job as complete with result. Returns True if successful."""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return False

            job.status = JobStatus.COMPLETE
            job.progress = 100
            job.result = result
            job.completed_at = datetime.utcnow()
            return True

    def fail_job(self, job_id: str, error: str) -> bool:
        """Mark job as failed with error message. Returns True if successful."""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return False

            job.status = JobStatus.FAILED
            job.error = error
            job.completed_at = datetime.utcnow()
            return True

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued or in-progress job. Returns True if successful."""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return False
            if job.status not in (JobStatus.QUEUED, JobStatus.IN_PROGRESS):
                return False

            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.utcnow()
            return True

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        limit: int = 100
    ) -> list[Dict[str, Any]]:
        """List jobs, optionally filtered by status."""
        with self.lock:
            jobs = list(self.jobs.values())
            if status:
                jobs = [j for j in jobs if j.status == status]

            # Sort by creation time, newest first
            jobs.sort(key=lambda j: j.created_at, reverse=True)

            return [j.to_dict() for j in jobs[:limit]]

    def cleanup_old_jobs(self) -> int:
        """Remove completed/failed jobs older than cleanup_interval. Returns count removed."""
        with self.lock:
            now = datetime.utcnow()
            to_remove = []

            for job_id, job in self.jobs.items():
                if job.status in (JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.CANCELLED):
                    if job.completed_at and (now - job.completed_at).seconds > self.cleanup_interval:
                        to_remove.append(job_id)

            for job_id in to_remove:
                del self.jobs[job_id]

            return len(to_remove)

    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        with self.lock:
            stats = {
                "total": len(self.jobs),
                "queued": sum(1 for j in self.jobs.values() if j.status == JobStatus.QUEUED),
                "in_progress": sum(1 for j in self.jobs.values() if j.status == JobStatus.IN_PROGRESS),
                "complete": sum(1 for j in self.jobs.values() if j.status == JobStatus.COMPLETE),
                "failed": sum(1 for j in self.jobs.values() if j.status == JobStatus.FAILED),
                "cancelled": sum(1 for j in self.jobs.values() if j.status == JobStatus.CANCELLED),
            }
            return stats


# Global queue instance (shared across API endpoints)
job_queue = JobQueue()
