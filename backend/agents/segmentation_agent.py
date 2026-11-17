"""Agent 2: Object Segmentation.

Finds the decorative design in the photo and isolates it from people, frames and
background. Uses the line-density method (see services/segmentation_service.py)
and, if available and in agreement, SAM 2. Never reports success for a mask it
could not validate: failures are returned with success=False and a warning.
"""
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from config import settings
from schemas.analysis import ImageAnalysis
from schemas.segmentation import SegmentationParams, SegmentationResult
from services import image_service
from services.segmentation_service import (
    Sam2Segmenter,
    Sam2UnavailableError,
    classical_segment,
    mask_bbox_px,
    mask_iou,
)

logger = logging.getLogger(__name__)

SAM2_MIN_IOU = 0.4  # SAM 2 must overlap the line-based region at least this much
SAM2_MAX_AREA_RATIO = 1.5  # ...and not be much larger (it tends to grab the whole glass pane)

_sam2: Sam2Segmenter | None = None


def get_sam2() -> Sam2Segmenter:
    """One shared SAM 2 wrapper so a loaded model is reused between jobs."""
    global _sam2
    if _sam2 is None:
        _sam2 = Sam2Segmenter(settings.sam2_checkpoint, settings.sam2_model_cfg, settings.sam2_device)
    return _sam2


def _save_png(array: np.ndarray, path: Path) -> None:
    Image.fromarray(array).save(path, format="PNG")  # PIL, so non-ASCII Windows paths are safe


class SegmentationAgent:
    name = "segmentation"

    def __init__(self, sam2: Sam2Segmenter | None = None) -> None:
        self._sam2 = sam2

    def run(
        self,
        job_id: str,
        image_path: Path,
        analysis: ImageAnalysis,
        params: SegmentationParams | None = None,
    ) -> SegmentationResult:
        rgb = np.asarray(image_service.load_oriented_image(image_path))
        h, w = rgb.shape[:2]
        params = params or SegmentationParams(sensitivity=analysis.extraction_hints.edge_sensitivity)
        backend = settings.segmentation_backend.strip().lower()

        out = classical_segment(rgb, analysis.bounding_box, params)
        mask, method = out.mask, ("classical" if out.success else "roi_box_fallback")
        confidence, warnings = out.confidence, list(out.warnings)
        if not out.success:
            warnings.append("Segmentation failed: the saved mask is only the search area, not the design.")
            logger.warning("Job %s: segmentation failed (%s)", job_id, "; ".join(out.warnings))

        sam2_status = "not_requested"
        if backend in ("auto", "sam2"):
            if not out.success:
                sam2_status = "not_run: no reliable region to prompt with"
            else:
                sam2_status, mask, method, confidence = self._try_sam2(rgb, out.mask, confidence, backend, warnings)

        success = out.success
        bbox_px = mask_bbox_px(mask)
        result = SegmentationResult(
            success=success,
            method=method,
            backend_requested=backend,
            sam2_status=sam2_status,
            confidence=round(confidence, 3),
            image_size=(w, h),
            mask_bbox=None if bbox_px is None else (bbox_px[0] / w, bbox_px[1] / h, bbox_px[2] / w, bbox_px[3] / h),
            roi_bbox=out.roi,
            area_fraction=round(float((mask > 0).sum()) / (w * h), 4),
            warnings=warnings,
        )
        result.files = self._save(job_id, rgb, mask, out.line_map, analysis, result, bbox_px)
        logger.info(
            "Job %s: segmentation %s via %s (confidence %.2f, area %.1f%%)",
            job_id, "ok" if success else "FAILED", method, result.confidence, result.area_fraction * 100,
        )
        return result

    # ------------------------------------------------------------------ SAM 2
    def _try_sam2(self, rgb, classical_mask, confidence, backend, warnings):
        sam = self._sam2 or get_sam2()
        ok, why = sam.availability()
        if not ok:
            if backend == "sam2":
                warnings.append(f"SAM 2 was requested but is unavailable ({why}); used the OpenCV method.")
            return f"unavailable: {why}", classical_mask, "classical", confidence
        h, w = classical_mask.shape
        x1, y1, x2, y2 = mask_bbox_px(classical_mask)
        pad_x, pad_y = int(0.05 * (x2 - x1)), int(0.05 * (y2 - y1))
        box = (max(0, x1 - pad_x), max(0, y1 - pad_y), min(w, x2 + pad_x), min(h, y2 + pad_y))
        try:
            sam_mask, _score = sam.segment(rgb, box)
        except Sam2UnavailableError as exc:
            warnings.append(f"SAM 2 could not run ({exc}); used the OpenCV method.")
            return f"unavailable: {exc}", classical_mask, "classical", confidence
        iou = mask_iou(sam_mask, classical_mask)
        ratio = float((sam_mask > 0).sum()) / max(float((classical_mask > 0).sum()), 1.0)
        if iou < SAM2_MIN_IOU or ratio > SAM2_MAX_AREA_RATIO:
            return (f"rejected: disagrees with line-based region (IoU {iou:.2f}, size x{ratio:.1f})",
                    classical_mask, "classical", confidence)
        return "used", sam_mask, "sam2", min(0.95, confidence + 0.05)

    # ------------------------------------------------------------------ output
    def _save(self, job_id, rgb, mask, line_map, analysis, result, bbox_px) -> dict[str, str]:
        out_dir = settings.output_dir / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        h, w = mask.shape

        _save_png(mask, out_dir / "mask.png")
        _save_png(line_map, out_dir / "line_map.png")
        isolated = np.where(mask[..., None] > 0, rgb, 255).astype(np.uint8)
        _save_png(isolated, out_dir / "isolated.png")
        if bbox_px:
            mx, my = int(0.03 * w), int(0.03 * h)
            x1, y1, x2, y2 = bbox_px
            crop = isolated[max(0, y1 - my) : min(h, y2 + my), max(0, x1 - mx) : min(w, x2 + mx)]
            _save_png(crop, out_dir / "isolated_crop.png")

        # debug overlay: red = AI box, yellow = search area, green = found region
        over = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
        t = max(2, int(0.003 * max(h, w)))

        def draw_box(b, color):
            if b:
                cv2.rectangle(over, (int(b[0] * w), int(b[1] * h)), (int(b[2] * w), int(b[3] * h)), color, t)

        draw_box(analysis.bounding_box, (0, 0, 255))
        draw_box(result.roi_bbox, (0, 220, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(over, contours, -1, (0, 200, 0) if result.success else (255, 0, 255), t)
        label = f"{result.method} ok={result.success} conf={result.confidence:.2f}"
        cv2.putText(over, label, (t * 3, t * 8), cv2.FONT_HERSHEY_SIMPLEX, max(0.6, h / 1400), (255, 255, 255), t + 2)
        cv2.putText(over, label, (t * 3, t * 8), cv2.FONT_HERSHEY_SIMPLEX, max(0.6, h / 1400), (0, 0, 0), t)
        _save_png(cv2.cvtColor(over, cv2.COLOR_BGR2RGB), out_dir / "debug_overlay.png")

        files = {n: n for n in ("mask.png", "isolated.png", "line_map.png", "debug_overlay.png")}
        if bbox_px:
            files["isolated_crop.png"] = "isolated_crop.png"
        (out_dir / "segmentation.json").write_text(result.model_copy(update={"files": files}).model_dump_json(indent=2), encoding="utf-8")
        files["segmentation.json"] = "segmentation.json"
        return files
