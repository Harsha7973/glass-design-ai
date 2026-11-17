"""Thread-safe in-memory job registry.

Deliberately simple (no Redis/Celery). Jobs are lost on restart; the files on
disk are removed by the age-based cleanup.
"""
import threading
from datetime import datetime, timezone

from schemas.job import JobEvent, JobResult, JobStatus


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, JobStatus] = {}
        self._lock = threading.Lock()

    def create(self, job_id: str, image_format: str, width: int, height: int) -> JobStatus:
        now = datetime.now(timezone.utc)
        job = JobStatus(
            job_id=job_id,
            status="uploaded",
            message="Upload validated and saved.",
            created_at=now,
            updated_at=now,
            image_format=image_format,
            width=width,
            height=height,
            events=[JobEvent(at=now, stage="uploaded", message="Upload validated and saved.")],
        )
        with self._lock:
            self._jobs[job_id] = job
        return job.model_copy(deep=True)

    def get(self, job_id: str) -> JobStatus | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.model_copy(deep=True) if job else None

    def update(
        self,
        job_id: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        message: str | None = None,
        error: str | None = None,
        result: JobResult | None = None,
        warning: str | None = None,
    ) -> None:
        """Record a real pipeline step. Unknown jobs are ignored (e.g. cleaned up)."""
        now = datetime.now(timezone.utc)
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if status is not None:
                job.status = status
            if stage is not None:
                job.stage = stage
            if message is not None:
                job.message = message
                job.events.append(JobEvent(at=now, stage=job.stage, message=message))
            if error is not None:
                job.error = error
            if result is not None:
                job.result = result
            if warning is not None and warning not in job.warnings:
                job.warnings.append(warning)
            job.updated_at = now


job_store = JobStore()
