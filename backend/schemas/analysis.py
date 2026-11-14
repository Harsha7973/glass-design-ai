"""Pydantic schemas for what Groq returns. Validation is deliberately lenient on
cosmetic fields (normalising odd values) but strict on the essentials."""
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Category = Literal["glass", "metal", "wood", "jewellery", "furniture", "stone", "fabric", "other"]
Sensitivity = Literal["low", "medium", "high"]
LineStyle = Literal["thin", "medium", "bold"]

BBox = tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max), each 0..1


def normalize_bbox(value) -> BBox | None:
    """Return a valid normalised box or None.

    Vision models sometimes answer on a 0-1000 grid instead of 0-1; values in
    (1.5, 1000] are therefore divided by 1000. This is a heuristic, so the
    segmentation stage must still treat the box as guidance, not truth.
    """
    if value is None:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    top = max(x1, y1, x2, y2)
    if top > 1000 or min(x1, y1, x2, y2) < 0:
        return None
    if top > 1.5:
        x1, y1, x2, y2 = (v / 1000 for v in (x1, y1, x2, y2))
    x1, y1, x2, y2 = (min(max(v, 0.0), 1.0) for v in (x1, y1, x2, y2))
    if x2 - x1 < 0.02 or y2 - y1 < 0.02:
        return None
    return (x1, y1, x2, y2)


class Candidate(BaseModel):
    description: str = ""
    bounding_box: BBox | None = None

    @field_validator("bounding_box", mode="before")
    @classmethod
    def _bbox(cls, v):
        return normalize_bbox(v)


class ExtractionHints(BaseModel):
    edge_sensitivity: Sensitivity = "medium"
    contrast_enhance: bool = True
    line_style: LineStyle = "medium"
    notes: str = ""

    @field_validator("edge_sensitivity", "line_style", mode="before")
    @classmethod
    def _lower(cls, v):
        return v.strip().lower() if isinstance(v, str) else v


class ImageAnalysis(BaseModel):
    """Agent 1 output (one Groq call per image)."""

    primary_object: str
    object_category: Category = "other"
    confidence: float = Field(ge=0.0, le=1.0)
    location: str = "center"
    bounding_box: BBox | None = None
    candidate_designs: list[Candidate] = Field(default_factory=list)
    unwanted_objects: list[str] = Field(default_factory=list)
    is_transparent_or_reflective: bool = False
    background_description: str = ""
    segmentation_required: bool = True
    outer_boundary_required: bool = True
    internal_details_required: bool = True
    extraction_hints: ExtractionHints = Field(default_factory=ExtractionHints)

    @field_validator("confidence", mode="before")
    @classmethod
    def _confidence(cls, v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise ValueError("confidence must be a number") from None
        if 1.0 < v <= 100.0:  # model answered as a percentage
            v = v / 100.0
        return min(max(v, 0.0), 1.0)

    @field_validator("object_category", mode="before")
    @classmethod
    def _category(cls, v):
        allowed = set(Category.__args__)
        v = v.strip().lower() if isinstance(v, str) else "other"
        return v if v in allowed else "other"

    @field_validator("location", mode="before")
    @classmethod
    def _location(cls, v):
        return v.strip().lower() if isinstance(v, str) and v.strip() else "center"

    @field_validator("bounding_box", mode="before")
    @classmethod
    def _bbox(cls, v):
        return normalize_bbox(v)

    @field_validator("unwanted_objects", "candidate_designs", mode="before")
    @classmethod
    def _none_to_list(cls, v):
        return [] if v is None else v

    @field_validator("extraction_hints", mode="before")
    @classmethod
    def _hints_none(cls, v):
        return {} if v is None else v

    @field_validator("primary_object")
    @classmethod
    def _non_empty(cls, v):
        if not v.strip():
            raise ValueError("primary_object is empty")
        return v.strip()


class PrimaryObject(BaseModel):
    name: str
    category: Category
    confidence: float
    location: str
    bounding_box: BBox | None
    multiple_candidates: bool


class ExtractionInstructions(BaseModel):
    """Parameters later phases use to steer segmentation and outline extraction."""

    region_of_interest: BBox | None
    segmentation_required: bool
    outer_boundary_required: bool
    internal_details_required: bool
    transparent_material: bool
    edge_sensitivity: Sensitivity
    contrast_enhance: bool
    line_style: LineStyle
    objects_to_suppress: list[str]
    notes: str


class GroqEvaluation(BaseModel):
    """Optional vision-based quality opinion (Phase 7). Advisory only."""

    quality_score: float = Field(ge=0.0, le=1.0)
    boundary_complete: bool = False
    internal_details_preserved: bool = False
    unwanted_elements_present: bool = False
    issues: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)

    @field_validator("quality_score", mode="before")
    @classmethod
    def _score(cls, v):
        v = float(v)
        if 1.0 < v <= 100.0:
            v = v / 100.0
        return min(max(v, 0.0), 1.0)
