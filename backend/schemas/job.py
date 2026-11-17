"""API response schemas."""
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

# status: where the job is overall; stage: which step is running right now.
Status = Literal["uploaded", "processing", "completed", "needs_review", "failed"]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    groq_configured: bool = False
    groq_model: str = ""


class JobEvent(BaseModel):
    """One real step the pipeline reported. These are the messages the UI shows."""

    at: datetime
    stage: str
    message: str


class JobResult(BaseModel):
    decision: Literal["accept", "needs_review", "failed"]
    score: float = Field(ge=0.0, le=1.0)
    attempts: int
    best_attempt: int
    summary: str
    problems: list[str] = Field(default_factory=list)
    primary_object: str = ""
    ai_analysis_used: bool = False
    files: dict[str, str] = Field(default_factory=dict)  # file name -> URL path


class JobStatus(BaseModel):
    job_id: str
    status: Status
    stage: str = "uploaded"  # uploaded | queued | analyzing | segmenting | extracting | evaluating | done
    message: str
    created_at: datetime
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    image_format: str
    width: int
    height: int
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    events: list[JobEvent] = Field(default_factory=list)
    result: JobResult | None = None
