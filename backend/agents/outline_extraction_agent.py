"""Agent 3: Outline Extraction.

Turns the isolated design into black lines on white: the complete outer boundary
plus the internal decorative lines. Writes PNGs for inspection and for the later
export/vector stage. Failure is reported (success=False), never hidden.
"""
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from config import settings
from schemas.analysis import ImageAnalysis
from schemas.outline import OutlineParams, OutlineResult
from services import image_service
from services.outline_service import extract_outline

logger = logging.getLogger(__name__)

_SENS = {"low": "low", "medium": "medium", "high": "high"}


def _black_on_white(ink: np.ndarray) -> np.ndarray:
    return (255 - (ink > 0).astype(np.uint8) * 255).astype(np.uint8)


def _save_png(array: np.ndarray, path: Path) -> None:
    Image.fromarray(array).save(path, format="PNG")


class OutlineExtractionAgent:
    name = "outline_extraction"

    def run(
        self,
        job_id: str,
        image_path: Path,
        mask: np.ndarray,
        analysis: ImageAnalysis | None = None,
        params: OutlineParams | None = None,
    ) -> OutlineResult:
        params = params or OutlineParams()
        if analysis is not None and analysis.extraction_hints is not None:
            hint = getattr(analysis.extraction_hints, "edge_sensitivity", None)
            if hint in _SENS and params.sensitivity == "medium":
                params = params.model_copy(update={"sensitivity": hint})
        rgb = np.asarray(image_service.load_oriented_image(image_path))
        out_dir = settings.output_dir / job_id
        out_dir.mkdir(parents=True, exist_ok=True)

        if mask is None or not (mask > 0).any():
            h, w = rgb.shape[:2]
            return OutlineResult(success=False, working_size=(w, h), crop_box_px=(0, 0, w, h),
                                 warnings=["No isolated region was provided, so no outline was attempted."],
                                 params_used=params)

        out = extract_outline(rgb, mask, params)
        files = {}
        for name, arr in (
            ("outer_boundary.png", _black_on_white(out.outer)),
            ("internal_design.png", _black_on_white(out.internal)),
            ("combined_outline.png", _black_on_white(out.combined)),
            ("centerlines.png", _black_on_white(out.centerlines)),
        ):
            _save_png(arr, out_dir / name)
            files[name] = name
        crop = out.crop_rgb
        _save_png(crop, out_dir / "crop.png")
        files["crop.png"] = "crop.png"
        side = np.concatenate([crop, np.repeat(_black_on_white(out.combined)[..., None], 3, axis=2)], axis=1)
        _save_png(side, out_dir / "outline_preview.png")
        files["outline_preview.png"] = "outline_preview.png"

        h, w = out.combined.shape
        result = OutlineResult(
            success=out.success,
            working_size=(w, h),
            crop_box_px=out.crop_box,
            components=out.components,
            total_line_length_px=out.total_length,
            noise_components_removed=out.removed,
            silhouette_area_fraction=round(float((out.silhouette > 0).mean()), 4),
            outer_contours=out.outer_contours,
            warnings=out.warnings,
            files=files,
            params_used=params,
        )
        (out_dir / "outline.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
        files["outline.json"] = "outline.json"
        return result
