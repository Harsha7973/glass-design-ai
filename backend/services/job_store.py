"""Thread-safe in-memory job registry.

Deliberately simple (no Redis/Celery). Jobs are lost on restart; the files on
disk are removed by the age-based cleanup.
"""
import threading
from datetime import datetime, timezone

from schemas.job import JobStatus


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, JobStatus] = {}
        self._lock = threading.Lock()

    def create(self, job_id: str, image_format: str, width: int, height: int) -> JobStatus:
        job = JobStatus(
            job_id=job_id,
            status="uploaded",
            message="Upload validated and saved.",
            created_at=datetime.now(timezone.utc),
            image_format=image_format,
            width=width,
            height=height,
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> JobStatus | None:
        with self._lock:
            return self._jobs.get(job_id)


job_store = JobStore()
