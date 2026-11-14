"""API response schemas."""
from datetime import datetime

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    groq_configured: bool = False
    groq_model: str = ""


class JobStatus(BaseModel):
    job_id: str
    status: str  # Phase 2 only has "uploaded"; later phases add pipeline stages
    message: str
    created_at: datetime
    image_format: str
    width: int
    height: int
