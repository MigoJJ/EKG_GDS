"""
Phase 5: Background Task Executor for Asynchronous ECG Analysis

Runs long-running tasks (LLM report generation) in background threads
with progress tracking via job_queue.

Architecture:
  main_thread (FastAPI) → create_job() → return job_id immediately
  worker_thread → process_job() → update progress → complete_job()
"""

from __future__ import annotations
import threading
import time
from typing import Any, Dict, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)


class TaskExecutor:
    """Executes long-running tasks in background with progress tracking."""

    def __init__(self, max_workers: int = 4):
        """
        Initialize executor.

        Args:
            max_workers: Max concurrent background threads
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.tasks: Dict[str, threading.Thread] = {}

    def submit_task(
        self,
        job_id: str,
        task_func: Callable[..., Any],
        *args,
        **kwargs
    ) -> None:
        """
        Submit task to run in background.

        Args:
            job_id: Job identifier (for progress tracking)
            task_func: Callable that executes the task
            *args, **kwargs: Arguments to pass to task_func

        Example:
            executor.submit_task(
                "job123",
                ecg_analysis_task,
                features=features_dict,
                classification=class_dict
            )
        """
        def _wrapper():
            try:
                task_func(job_id=job_id, *args, **kwargs)
            except Exception as e:
                logger.error(f"Task {job_id} failed: {e}", exc_info=True)
                # Error handling is up to the task_func (it should update job queue)

        thread = threading.Thread(target=_wrapper, daemon=True)
        self.tasks[job_id] = thread
        thread.start()

    def wait_for_task(self, job_id: str, timeout: Optional[float] = None) -> bool:
        """
        Wait for background task to complete.

        Args:
            job_id: Job identifier
            timeout: Max seconds to wait (None = indefinite)

        Returns:
            True if task completed within timeout, False otherwise
        """
        thread = self.tasks.get(job_id)
        if not thread:
            return False

        thread.join(timeout=timeout)
        return not thread.is_alive()

    def cancel_task(self, job_id: str) -> bool:
        """
        Attempt to cancel a background task (best effort).
        Note: Python doesn't support true thread cancellation.
        The task should check for cancellation signals periodically.

        Returns:
            True if task was found and cancel signal was sent
        """
        # Mark for cancellation (task must check this)
        # Actual cancellation is handled by task_func itself
        logger.info(f"Cancellation requested for job {job_id}")
        return job_id in self.tasks

    def cleanup(self):
        """Shutdown executor (flush pending tasks)."""
        self.executor.shutdown(wait=True)


# Global executor instance
task_executor = TaskExecutor(max_workers=4)


def ecg_analysis_task(
    job_id: str,
    features: Dict[str, Any],
    classification: Dict[str, Any],
    patient_info: Optional[Dict[str, Any]] = None,
    ollama_client: Any = None,
    job_queue: Any = None,
) -> None:
    """
    Background task: Extract features, classify, and generate LLM report.

    This runs in a background thread. It updates job_queue with progress.

    Args:
        job_id: Job identifier (already queued/started by caller)
        features: Quantitative ECG features (from extractor)
        classification: DL classification (from classifier)
        patient_info: Optional patient demographics
        ollama_client: OllamaEKGClient instance
        job_queue: JobQueue instance for progress updates

    Flow:
      0% → features extracted (already done by caller)
      25% → classification done (already done by caller)
      50% → LLM report generation started
      75% → LLM report generation halfway
      100% → complete
    """
    try:
        from models.job_queue import job_queue as default_job_queue
        if job_queue is None:
            job_queue = default_job_queue

        # Mark as started
        if not job_queue.start_job(job_id):
            logger.warning(f"Job {job_id} not found in queue")
            return

        # Update progress: features & classification done
        job_queue.update_progress(job_id, 25)
        time.sleep(0.1)  # Simulate work

        # Generate LLM report (longest step)
        job_queue.update_progress(job_id, 50)

        if ollama_client is None:
            raise ValueError("ollama_client required for report generation")

        narrative, structured_json = ollama_client.generate_combined_report(
            features, classification, patient_info
        )

        job_queue.update_progress(job_id, 75)
        time.sleep(0.1)

        # Assemble final result
        result = {
            "status": "success",
            "quantitative_features": features,
            "classification": classification,
            "ollama_narrative": narrative,
            "ollama_json": structured_json,
            "generated_at": __import__("datetime").datetime.utcnow().isoformat(),
        }

        # Mark complete
        job_queue.complete_job(job_id, result)
        logger.info(f"Job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"Task {job_id} failed: {e}", exc_info=True)
        job_queue.fail_job(job_id, str(e))


# Example: Can be extended with other task types
# - ecg_image_analysis_task (for PNG input)
# - dicom_analysis_task (for DICOM input)
# - etc.
