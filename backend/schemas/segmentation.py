"""Schemas for Agent 2 (segmentation)."""
from typing import Literal

from pydantic import BaseModel, Field

BBox = tuple[float, float, float, float]  # normalised (x_min, y_min, x_max, y_max)

Method = Literal["classical", "sam2", "roi_box_fallback"]


class SegmentationParams(BaseModel):
    """Tunable knobs. The quality agent (Phase 7) adjusts these on retries."""

    working_max_side: int = Field(default=1600, ge=400)  # analysis resolution
    roi_expand: float = Field(default=0.5, ge=0.0, le=3.0)  # box margin, fraction of box size
    roi_min_margin: float = Field(default=0.08, ge=0.0, le=0.5)  # box margin floor (image fraction)
    line_scale: float = Field(default=0.010, ge=0.003, le=0.05)  # thin-line width scale
    density_sigma: float = Field(default=0.015, ge=0.004, le=0.08)
    density_scale: float = Field(default=1.0, ge=0.3, le=3.0)  # >1 = stricter region
    grow_reach: float = Field(default=0.12, ge=0.0, le=0.4)  # how far faint lines may extend the region
    sensitivity: Literal["low", "medium", "high"] = "medium"
    min_area_fraction: float = 0.003
    max_area_fraction: float = 0.70


class SegmentationResult(BaseModel):
    """What the segmentation stage found. `success` is False when no trustworthy
    mask was produced; the mask file is then only a placeholder region."""

    success: bool
    method: Method
    backend_requested: str
    sam2_status: str  # not_requested | unavailable: <why> | used | rejected: <why>
    confidence: float = Field(ge=0.0, le=1.0)  # heuristic, NOT a guarantee
    image_size: tuple[int, int]  # (width, height)
    mask_bbox: BBox | None = None
    roi_bbox: BBox | None = None
    area_fraction: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    files: dict[str, str] = Field(default_factory=dict)
