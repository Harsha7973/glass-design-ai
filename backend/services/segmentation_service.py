"""Local computer-vision segmentation.

Why not plain colour/GrabCut segmentation: etched or frosted glass has almost the
same colour as its background, so "foreground vs background" colour models fail.
What does stand out is a concentration of THIN LINES (the etching). The classical
backend therefore:

1. enhances contrast (CLAHE) and softens frost grain,
2. extracts thin bright/dark structures with morphological top-hat/black-hat
   (wide structures such as door frames and clothing are ignored),
3. drops tiny speckles (connected-component size filter),
4. turns line density into a region, and keeps the region that holds the most
   line pixels inside the search area given by the (unreliable) Groq box.

The Groq box only chooses WHICH region; the region may extend beyond the box.
SAM 2 is an optional second opinion and must agree with this result.
"""
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from schemas.segmentation import BBox, SegmentationParams

logger = logging.getLogger(__name__)

SENSITIVITY_FACTOR = {"low": 1.25, "medium": 1.0, "high": 0.8}  # multiplies the line threshold


def _odd(n: float) -> int:
    n = max(3, int(round(n)))
    return n if n % 2 else n + 1


def expand_roi(bbox: BBox | None, params: SegmentationParams) -> BBox | None:
    """Grow the Groq box into a search area (still normalised 0..1)."""
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    mx = max((x2 - x1) * params.roi_expand, params.roi_min_margin)
    my = max((y2 - y1) * params.roi_expand, params.roi_min_margin)
    return (max(0.0, x1 - mx), max(0.0, y1 - my), min(1.0, x2 + mx), min(1.0, y2 + my))


@dataclass
class ClassicalOutput:
    success: bool
    mask: np.ndarray  # uint8 0/255, full image size
    line_map: np.ndarray  # uint8 0/255, full image size (debug)
    roi: BBox | None
    confidence: float
    warnings: list[str] = field(default_factory=list)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def classical_segment(
    rgb: np.ndarray, bbox: BBox | None, params: SegmentationParams | None = None
) -> ClassicalOutput:
    params = params or SegmentationParams()
    h, w = rgb.shape[:2]
    scale = min(1.0, params.working_max_side / max(h, w))
    small = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    sh, sw = small.shape[:2]
    m = max(sh, sw)

    # 1. contrast + soften frost grain
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(0.8, 0.0012 * m))
    eq = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)

    # 2. thin-line response (bright OR dark thin structures)
    k = _odd(max(7, params.line_scale * m))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    line = np.maximum(
        cv2.morphologyEx(eq, cv2.MORPH_TOPHAT, kernel), cv2.morphologyEx(eq, cv2.MORPH_BLACKHAT, kernel)
    ).astype(np.float32)
    line = np.clip(line / max(float(np.percentile(line, 99.5)), 1.0), 0, 1)
    line_u8 = (line * 255).astype(np.uint8)
    otsu, _ = cv2.threshold(line_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = (line_u8 > otsu * SENSITIVITY_FACTOR[params.sensitivity]).astype(np.uint8)

    # 3. drop speckles
    min_area = max(12, int(0.00004 * sh * sw))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    binary = keep[labels].astype(np.uint8)

    # 4. line density -> region
    density = cv2.GaussianBlur(binary.astype(np.float32), (0, 0), sigmaX=params.density_sigma * m)
    dmax = float(density.max())
    warnings: list[str] = []

    def finish(success, region_small, roi, confidence):
        mask_small = region_small.astype(np.uint8)
        mask_full = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_LINEAR) if scale < 1 else mask_small
        mask_full = ((mask_full > 127) * 255).astype(np.uint8)
        lm = cv2.resize(binary * 255, (w, h), interpolation=cv2.INTER_NEAREST) if scale < 1 else binary * 255
        return ClassicalOutput(success, mask_full, lm.astype(np.uint8), roi, confidence, warnings)

    roi = expand_roi(bbox, params)
    roi_px = None
    if roi is not None:
        roi_px = (int(roi[0] * sw), int(roi[1] * sh), max(int(roi[2] * sw), int(roi[0] * sw) + 1), max(int(roi[3] * sh), int(roi[1] * sh) + 1))

    def fallback(reason: str):
        warnings.append(reason)
        region = np.zeros((sh, sw), np.uint8)
        if roi_px is not None:
            region[roi_px[1] : roi_px[3], roi_px[0] : roi_px[2]] = 255
        else:
            region[:] = 255
        return finish(False, region, roi, 0.0)

    if dmax < 1e-4:
        return fallback("No line structure found in the image.")

    d_u8 = (density / dmax * 255).astype(np.uint8)
    d_thr, _ = cv2.threshold(d_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    region = (d_u8 > d_thr * params.density_scale).astype(np.uint8) * 255
    close_k = _odd(0.02 * m)
    region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)))
    region = _fill_holes(region)

    # choose the region holding the most line pixels inside the search area
    n, labels, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
    if n <= 1:
        return fallback("No dense line region found.")
    in_roi = np.zeros((sh, sw), bool)
    if roi_px is not None:
        in_roi[roi_px[1] : roi_px[3], roi_px[0] : roi_px[2]] = True
    else:
        in_roi[:] = True
    scores = np.bincount(labels[in_roi & (binary > 0)].ravel(), minlength=n).astype(float)
    scores[0] = 0
    best = int(scores.argmax())
    if scores[best] <= 0:
        return fallback("No line structure inside the search area.")
    chosen = (labels == best).astype(np.uint8) * 255

    # grow into fainter line areas (e.g. a faint pointed tip) that connect to the core,
    # but only within a limited distance so the region cannot leak onto other objects
    low = (d_u8 > d_thr * params.density_scale * 0.5).astype(np.uint8) * 255
    low = cv2.morphologyEx(low, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)))
    _, low_labels = cv2.connectedComponents(low, connectivity=8)
    touched = np.unique(low_labels[chosen > 0])
    grown = np.isin(low_labels, touched[touched > 0])
    reach = cv2.dilate(chosen, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(params.grow_reach * m),) * 2))
    chosen = _fill_holes(((grown & (reach > 0)) | (chosen > 0)).astype(np.uint8) * 255)

    # 5. sanity checks (a bad mask must not be reported as success)
    area_frac = float((chosen > 0).sum()) / (sh * sw)
    if area_frac < params.min_area_fraction:
        return fallback(f"Region too small ({area_frac:.2%} of image).")
    if area_frac > params.max_area_fraction:
        return fallback(f"Region covers {area_frac:.0%} of the image; likely texture, not a single design.")

    # heuristic confidence: how much of the found line mass sits in the chosen region
    total_lines = float(binary.sum()) or 1.0
    share = float(((chosen > 0) & (binary > 0)).sum()) / total_lines
    cover = float(scores[best]) / max(float((binary[in_roi] > 0).sum()), 1.0)
    confidence = float(np.clip(0.5 * share + 0.5 * cover, 0.05, 0.9))
    if roi is None:
        warnings.append("No box from the AI; chose the densest line region in the whole image.")
    return finish(True, chosen, roi, confidence)


# ---------------------------------------------------------------- SAM 2
class Sam2UnavailableError(RuntimeError):
    """SAM 2 cannot be used (package, weights or model config missing)."""


class Sam2Segmenter:
    """Optional adapter around Meta's SAM 2 image predictor (box prompt).

    Written against SAM 2's documented API (build_sam2, SAM2ImagePredictor,
    set_image, predict). It has NOT been run against the real model in this
    project's test environment; tests use a stand-in predictor.
    """

    def __init__(self, checkpoint: str, model_cfg: str, device: str = "cpu", predictor=None) -> None:
        self.checkpoint = checkpoint
        self.model_cfg = model_cfg
        self.device = device
        self._predictor = predictor
        self._lock = threading.Lock()

    def availability(self) -> tuple[bool, str]:
        if self._predictor is not None:
            return True, "ready"
        if not self.checkpoint:
            return False, "SAM2_CHECKPOINT is not set"
        if not Path(self.checkpoint).is_file():
            return False, f"checkpoint file not found: {self.checkpoint}"
        try:
            import torch  # noqa: F401
            import sam2  # noqa: F401
        except ImportError as exc:
            return False, f"package not installed ({exc.name})"
        return True, "ready"

    def _load(self):
        with self._lock:
            if self._predictor is None:  # reuse the loaded model across jobs
                ok, why = self.availability()
                if not ok:
                    raise Sam2UnavailableError(why)
                try:
                    from sam2.build_sam import build_sam2
                    from sam2.sam2_image_predictor import SAM2ImagePredictor

                    model = build_sam2(self.model_cfg, self.checkpoint, device=self.device)
                    self._predictor = SAM2ImagePredictor(model)
                except Exception as exc:  # noqa: BLE001 - third-party init can fail in many ways
                    raise Sam2UnavailableError(f"could not initialise SAM 2 ({type(exc).__name__})") from None
                logger.info("SAM 2 loaded on %s", self.device)
            return self._predictor

    def segment(self, rgb: np.ndarray, box_px: tuple[int, int, int, int]) -> tuple[np.ndarray, float]:
        """Return (mask uint8 0/255, model score) for a pixel box prompt."""
        predictor = self._load()
        try:
            import torch

            ctx = torch.inference_mode()
        except ImportError:  # stand-in predictors in tests
            from contextlib import nullcontext

            ctx = nullcontext()
        with ctx:
            predictor.set_image(rgb)
            masks, scores, _ = predictor.predict(box=np.array(box_px)[None, :], multimask_output=True)
        i = int(np.argmax(scores))
        return ((np.asarray(masks[i]) > 0) * 255).astype(np.uint8), float(scores[i])


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a > 0, b > 0).sum())
    union = float(np.logical_or(a > 0, b > 0).sum())
    return inter / union if union else 0.0


def mask_bbox_px(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
