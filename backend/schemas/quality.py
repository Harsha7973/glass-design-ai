"""Schemas for Agent 4 (quality evaluation and the retry loop)."""
from typing import Literal

from pydantic import BaseModel, Field

from schemas.analysis import GroqEvaluation
from schemas.outline import OutlineParams
from schemas.segmentation import SegmentationParams

ACCEPT_SCORE = 0.80  # at or above: good enough, stop (clean synthetic drawings score ~0.99)
MIN_USABLE_SCORE = 0.45  # below this even the best attempt is reported as failed

Decision = Literal["accept", "retry", "needs_review", "failed"]


class QualityMetrics(BaseModel):
    """Local, deterministic measurements (no AI). All values 0..1, higher is better."""

    contrast: float  # drawn lines that stand out clearly from the frost texture
    recall: float  # long photo edges inside the design that have a line
    boundary_fit: float  # outer boundary that follows actual strokes (low = blobby)
    continuity: float  # line length in reasonably long strokes (low = fragmentary)
    clutter: float  # 1 = few stroke ends per length (clean); low = tangled, noisy network
    line_density: float  # fraction of the design area covered by lines (info only)


class QualityReport(BaseModel):
    score: float = Field(ge=0.0, le=1.0)  # local score, optionally blended with Groq's
    local_score: float = Field(ge=0.0, le=1.0)
    decision: Decision
    metrics: QualityMetrics | None = None
    problems: list[str] = Field(default_factory=list)  # plain-language findings
    groq: GroqEvaluation | None = None
    groq_status: str = "not_used"  # not_used | used | unavailable: <why>


class AttemptRecord(BaseModel):
    attempt: int
    outline_params: OutlineParams
    segmentation_params: SegmentationParams
    report: QualityReport


class RefinementResult(BaseModel):
    """Outcome of the capped retry loop. `best_attempt` is what later phases export."""

    decision: Decision
    best_attempt: int
    attempts: list[AttemptRecord]
    max_attempts: int
    message: str
    files: dict[str, str] = Field(default_factory=dict)
