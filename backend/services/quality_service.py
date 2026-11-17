"""Deterministic quality measurements for an extracted outline, plus the rules that
turn a diagnosis into new parameters for the next attempt. No AI involved here."""
import cv2
import numpy as np

from schemas.outline import OutlineParams
from services.outline_service import find_endpoints
from schemas.quality import QualityMetrics
from schemas.segmentation import SegmentationParams

LEVELS = ["low", "medium", "high"]


def _ell(r: float) -> np.ndarray:
    k = max(1, int(round(r)) * 2 + 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _long_edges(gray: np.ndarray, min_len: int) -> np.ndarray:
    """Canny edges with short specks removed (frost grain makes lots of tiny edges)."""
    smooth = cv2.bilateralFilter(gray, 7, 35, 5)
    med = float(np.median(smooth))
    edges = cv2.Canny(smooth, int(max(10, 0.33 * med)), int(max(30, 0.66 * med)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(edges, _ell(1)), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_len * 2
    return (edges > 0) & keep[lab]


def measure(crop_rgb: np.ndarray, centerlines: np.ndarray, outer_ink: np.ndarray) -> QualityMetrics:
    """All inputs share one size. centerlines/outer_ink are 0/255 (255 = line)."""
    h, w = centerlines.shape
    m = max(h, w)
    lines = centerlines > 0
    total = int(lines.sum())
    if total == 0:
        return QualityMetrics(contrast=0, recall=0, boundary_fit=0, continuity=0, clutter=0, line_density=0)

    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    if gray.shape != lines.shape:
        gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
    gf = gray.astype(np.float32)

    # contrast: local contrast under each line vs the frost texture far from any line
    local = cv2.GaussianBlur(gf, (0, 0), 0.006 * m)
    c_map = cv2.dilate(np.abs(cv2.GaussianBlur(gf, (0, 0), 1.2) - local), np.ones((3, 3), np.uint8))
    far = cv2.dilate(lines.astype(np.uint8), _ell(0.03 * m)) == 0
    bg = c_map[far]
    thr = float(np.percentile(bg, 90)) if bg.size > 50 else float(np.percentile(c_map, 50))
    contrast = float((c_map[lines] > thr).mean())

    edges = _long_edges(gray, max(8, int(0.012 * m)))
    tol = max(2, int(0.004 * m))
    near_line = cv2.dilate(lines.astype(np.uint8), _ell(tol + 1)) > 0

    cnts, _ = cv2.findContours((outer_ink > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    area = np.zeros((h, w), np.uint8)
    cv2.drawContours(area, cnts, -1, 1, cv2.FILLED)
    inside = area > 0
    n_edge = int((edges & inside).sum())
    recall = float((edges & inside & near_line).sum() / n_edge) if n_edge else 0.0

    thin = np.zeros((h, w), np.uint8)
    cv2.drawContours(thin, cnts, -1, 1, 1)
    near_stroke = cv2.dilate(lines.astype(np.uint8), _ell(max(2, 0.003 * m))) > 0  # tight: really on a stroke
    n_thin = int(thin.sum())
    boundary_fit = float(((thin > 0) & near_stroke).sum() / n_thin) if n_thin else 0.0

    n, _, stats, _ = cv2.connectedComponentsWithStats(lines.astype(np.uint8), connectivity=8)
    sizes = stats[1:, cv2.CC_STAT_AREA]
    continuity = float(sizes[sizes >= 0.06 * m].sum() / max(sizes.sum(), 1))

    ends_per_k = 1000.0 * int(find_endpoints(lines).sum()) / total
    clutter = 1.0 - (ends_per_k - 500.0) / 1500.0  # ~500/1k on a clean drawing, ~1850 on frosted photo

    density = float(total / max(int(inside.sum()), 1))
    c = lambda v: float(min(max(v, 0.0), 1.0))  # noqa: E731
    return QualityMetrics(contrast=c(contrast), recall=c(recall), boundary_fit=c(boundary_fit),
                          continuity=c(continuity), clutter=c(clutter), line_density=c(density))


def local_score(mt: QualityMetrics) -> float:
    contrast = c01((mt.contrast - 0.6) / 0.4)  # 60% = no better than texture, 100% = clearly real
    recall = c01(mt.recall / 0.7)
    return round(0.25 * contrast + 0.20 * recall + 0.20 * mt.boundary_fit + 0.15 * mt.continuity + 0.20 * mt.clutter, 4)


def c01(v: float) -> float:
    return float(min(max(v, 0.0), 1.0))


def diagnose(mt: QualityMetrics) -> list[str]:
    """Plain-language problems, ordered by how much they hurt."""
    p = []
    if mt.contrast < 0.8:
        p.append("many drawn lines barely stand out from the frost texture (probably frost or reflections)")
    if mt.clutter < 0.5:
        p.append("the lines form a tangled network with many loose ends (typical of frost noise)")
    if mt.recall < 0.35:
        p.append("many clear edges in the photo have no line (detail is missing)")
    if mt.boundary_fit < 0.5:
        p.append("the outer boundary does not follow the actual strokes (blobby or wandering)")
    if mt.continuity < 0.6:
        p.append("the lines are fragmentary (many short pieces)")
    return p


def next_params(op: OutlineParams, sp: SegmentationParams, mt: QualityMetrics | None, groq_unwanted: bool = False):
    """Rule-based adjustment for the next attempt. Returns (outline_params, seg_params, note)."""
    if mt is None:
        return op, sp.model_copy(update={"density_scale": min(sp.density_scale * 1.2, 3.0)}), "retry with a stricter region"
    lvl = LEVELS.index(op.sensitivity)
    noisy = mt.contrast < 0.8 or mt.clutter < 0.5 or groq_unwanted
    sparse = mt.recall < 0.35 and not noisy
    ou, su, notes = {}, {}, []
    if noisy:
        if mt.contrast < 0.8 and lvl > 0:
            ou["sensitivity"] = LEVELS[lvl - 1]
            notes.append(f"lower sensitivity to {LEVELS[lvl - 1]}")
        if mt.clutter < 0.5:  # tangled: usually doubled bevel lines, so merge them more
            ou["merge_sigma"] = min(op.merge_sigma * 1.6, 0.008)
            notes.append("merge paired/double lines more strongly")
        ou["min_component_len"] = min(op.min_component_len * 1.5, 0.2)
        ou["isolation_dist"] = max(op.isolation_dist * 0.7, 0.01)
        su["density_scale"] = min(sp.density_scale * 1.25, 3.0)
        su["grow_reach"] = max(sp.grow_reach * 0.6, 0.0)
        notes.append("remove short/isolated strokes and tighten the region")
    elif sparse and op.bright_lines_only:
        ou["bright_lines_only"] = False  # dark-engraved glass: lines are not bright, so accept both
        notes.append("also accept dark lines")
    elif sparse and lvl < 2:
        ou["sensitivity"] = LEVELS[lvl + 1]
        notes.append(f"raise sensitivity to {LEVELS[lvl + 1]}")
    if mt.boundary_fit < 0.5:
        ou["gap_close"] = max(op.gap_close * 0.5, 0.0)
        notes.append("hug the strokes more tightly at the boundary")
    if mt.continuity < 0.6:
        ou["bridge_dist"] = min(op.bridge_dist * 1.5, 0.2)
        notes.append("join broken stroke ends over longer gaps")
    return (op.model_copy(update=ou), sp.model_copy(update=su), "; ".join(notes) or "no change possible")
