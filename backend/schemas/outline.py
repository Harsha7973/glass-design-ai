"""Schemas for Agent 3 (outline extraction)."""
from typing import Literal

from pydantic import BaseModel, Field

Level = Literal["low", "medium", "high"]
Style = Literal["thin", "medium", "bold"]

# Sensitivity: a stroke counts as "strong" when its evidence reaches this fraction of the
# strongest strokes in the design (99th percentile). Lower = more sensitive = fainter
# strokes are kept, and more noise gets through.
HIGH_FRAC = {"low": 0.55, "medium": 0.40, "high": 0.30}
CONTRAST_C = {"low": 10.0, "medium": 7.0, "high": 5.0}  # grey levels, local-contrast gate
THICKNESS = {"thin": 0.0016, "medium": 0.0026, "bold": 0.0040}  # fraction of longest side


class OutlineParams(BaseModel):
    """Tunable knobs. The quality agent (Phase 7) adjusts these on retries."""

    sensitivity: Level = "low"  # strict by default: frost speckle is worse than a few missing faint strokes
    line_style: Style = "medium"
    contrast_enhance: bool = True
    bright_lines_only: bool = True  # etched grooves that scatter light; ignores dark frost speckle
    min_side_px: int = Field(default=1200, ge=400)  # small crops are upscaled to this
    max_side_px: int = Field(default=2400, ge=800)  # huge crops are downscaled to this
    crop_margin: float = Field(default=0.04, ge=0.0, le=0.3)
    # stroke scales as fractions of the working size (1200 px -> sigma 2.2 / 3.4 / 4.8)
    ridge_scales: tuple[float, ...] = (0.0012, 0.0019, 0.0028)
    merge_sigma: float = Field(default=0.0018, ge=0.001, le=0.02)  # merges bright/dark stroke pairs
    bridge_dist: float = Field(default=0.02, ge=0.0, le=0.2)  # max gap joined between aligned stroke ends
    low_ratio: float = Field(default=0.5, ge=0.2, le=0.9)  # weak threshold / strong threshold
    min_component_len: float = Field(default=0.014, ge=0.002, le=0.2)  # fraction of size
    anchor_len: float = Field(default=0.08, ge=0.02, le=0.5)  # "main design" stroke length
    isolation_dist: float = Field(default=0.035, ge=0.005, le=0.3)  # weak strokes must be near an anchor
    gap_close: float = Field(default=0.003, ge=0.0, le=0.05)  # silhouette gap bridging
    min_total_len: float = Field(default=0.6, ge=0.1, le=5.0)  # below this much line, report failure
    spur_len: float = Field(default=0.018, ge=0.0, le=0.1)  # dead-end branches shorter than this are pruned
    boundary_band: float = Field(default=0.004, ge=0.0, le=0.03)  # keep interior lines this far from the border


class OutlineResult(BaseModel):
    """What the outline stage found. `success` is False when too little reliable
    line structure was found; files are then still written for inspection."""

    success: bool
    working_size: tuple[int, int]  # (width, height) of the output images
    crop_box_px: tuple[int, int, int, int]  # crop in the ORIGINAL image (x1, y1, x2, y2)
    components: int = 0
    total_line_length_px: int = 0
    noise_components_removed: int = 0
    silhouette_area_fraction: float = 0.0  # of the output image
    outer_contours: int = 0
    warnings: list[str] = Field(default_factory=list)
    files: dict[str, str] = Field(default_factory=dict)
    params_used: OutlineParams | None = None
