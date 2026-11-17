"""Outline extraction: turns an isolated design into clean black lines.

Why not plain Canny: etched glass strokes are thin bright bands with a shadowed
side. Canny returns BOTH sides of every band (double lines) and fires on every frost
grain. This pipeline instead
  1. flattens the frost grain (bilateral filter), boosts local contrast (CLAHE) and
     sharpens (unsharp mask),
  2. measures how "line-like" every pixel is at several stroke widths (Hessian ridge
     response) and keeps only the centre of each stroke (non-maximum suppression),
  3. keeps strokes whose strength is statistically far above the background noise
     (MAD-based thresholds + hysteresis), and only where there is real local contrast
     (adaptive-threshold gate),
  4. removes specks, short strokes without Canny edge support, and short strokes that
     are not near the main design (connected-component analysis),
  5. builds the design silhouette from the surviving strokes (morphology) and traces
     its outer boundary (contours) separately from the interior pattern.
"""
import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from schemas.outline import CONTRAST_C, HIGH_FRAC, THICKNESS, OutlineParams

logger = logging.getLogger(__name__)


def _odd(n: float, minimum: int = 3) -> int:
    n = max(minimum, int(round(n)))
    return n if n % 2 else n + 1


def _ellipse(radius: float) -> np.ndarray:
    r = max(1, int(round(radius)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


# ------------------------------------------------------------ preprocessing
def prepare_crop(rgb: np.ndarray, mask: np.ndarray, params: OutlineParams):
    """Crop to the design (plus margin) and rescale to a sensible working size.

    Returns (crop_rgb, crop_mask, crop_box_in_original, scale)."""
    ys, xs = np.where(mask > 0)
    h, w = mask.shape
    if xs.size == 0:
        raise ValueError("empty mask")
    x1, x2, y1, y2 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    mx, my = int((x2 - x1) * params.crop_margin) + 2, int((y2 - y1) * params.crop_margin) + 2
    x1, y1, x2, y2 = max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my)
    crop, cmask = rgb[y1:y2, x1:x2], mask[y1:y2, x1:x2]
    longest = max(crop.shape[:2])
    scale = 1.0
    if longest < params.min_side_px:
        scale = params.min_side_px / longest
    elif longest > params.max_side_px:
        scale = params.max_side_px / longest
    if scale != 1.0:
        size = (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale)))
        interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        crop = cv2.resize(crop, size, interpolation=interp)
        cmask = cv2.resize(cmask, size, interpolation=cv2.INTER_LINEAR)
        cmask = ((cmask > 127) * 255).astype(np.uint8)
    return crop, cmask, (x1, y1, x2, y2), scale


def enhance(crop_rgb: np.ndarray, params: OutlineParams) -> tuple[np.ndarray, np.ndarray]:
    """Return (enhanced_gray_uint8, smoothed_gray_uint8_for_canny)."""
    m = max(crop_rgb.shape[:2])
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    # bilateral: smooths frost grain but keeps stroke edges
    smooth = cv2.bilateralFilter(gray, d=7, sigmaColor=35, sigmaSpace=max(3.0, 0.004 * m))
    out = smooth
    if params.contrast_enhance:
        out = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(out)
    # unsharp mask: sharpen stroke edges
    blur = cv2.GaussianBlur(out, (0, 0), sigmaX=max(1.0, 0.002 * m))
    out = cv2.addWeighted(out, 1.6, blur, -0.6, 0)
    return out, smooth


# ----------------------------------------------------------- ridge detection
def ridge_response(gray: np.ndarray, scales_px: list[float], bright_only: bool = False):
    """Multi-scale Hessian line response for bright ridges AND dark valleys.

    Returns (response float32, normal_angle float32). The angle points ACROSS the
    stroke, which is the direction non-maximum suppression compares along."""
    g0 = gray.astype(np.float32)
    best = np.zeros_like(g0)
    angle = np.zeros_like(g0)
    for s in scales_px:
        g = cv2.GaussianBlur(g0, (0, 0), s)
        ixx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
        iyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
        ixy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
        half, diff = (ixx + iyy) / 2, (ixx - iyy) / 2
        root = np.sqrt(diff * diff + ixy * ixy)
        l1, l2 = half + root, half - root  # l1 >= l2
        bright = np.maximum(-l2 - np.abs(l1), 0)  # strong negative curvature across, flat along
        dark = np.maximum(l1 - np.abs(l2), 0)
        if bright_only:
            dark = np.zeros_like(dark)
        resp = np.maximum(bright, dark) * (s * s)  # scale-normalised
        theta1 = 0.5 * np.arctan2(2 * ixy, ixx - iyy)  # direction of the larger eigenvalue
        normal = np.where(bright >= dark, theta1 + np.pi / 2, theta1)
        take = resp > best
        best = np.where(take, resp, best)
        angle = np.where(take, normal, angle)
    return best, angle


def non_max_suppress(resp: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Keep only pixels that are local maxima across the stroke (1-px centre lines)."""
    h, w = resp.shape
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    nx, ny = np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)
    a = cv2.remap(resp, xx + nx, yy + ny, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    b = cv2.remap(resp, xx - nx, yy - ny, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return (resp >= a) & (resp >= b)


def contrast_gate(gray: np.ndarray, c_value: float) -> np.ndarray:
    """Adaptive-threshold map of pixels that differ from their neighbourhood mean."""
    block = _odd(0.015 * max(gray.shape), 15)
    dark = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, c_value)
    light = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, -c_value)
    return cv2.dilate(cv2.bitwise_or(dark, light), _ellipse(2)) > 0


def hysteresis(nms: np.ndarray, resp: np.ndarray, low: float, high: float) -> np.ndarray:
    """Keep weak centre-line pixels only where they connect to a strong one."""
    weak = nms & (resp > low)
    strong = nms & (resp > high)
    if not strong.any():
        return np.zeros_like(weak)
    bridged = cv2.dilate(weak.astype(np.uint8), np.ones((3, 3), np.uint8))  # joins 1-px breaks
    n, labels = cv2.connectedComponents(bridged, connectivity=8)
    seeds = np.unique(labels[strong])
    keep = np.isin(labels, seeds[seeds > 0])
    return keep & weak


def _component_stats(lines: np.ndarray):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(lines.astype(np.uint8), connectivity=8)
    return n, labels, stats[:, cv2.CC_STAT_AREA].astype(np.int64)


def clean_lines(lines: np.ndarray, canny: np.ndarray, params: OutlineParams) -> tuple[np.ndarray, int]:
    """Drop specks, unsupported short strokes and isolated short strokes."""
    m = max(lines.shape)
    min_len = max(4, int(params.min_component_len * m))
    anchor_len = max(min_len * 2, int(params.anchor_len * m))
    removed = 0

    n, labels, length = _component_stats(lines)
    keep = np.zeros(n, bool)
    keep[1:] = length[1:] >= min_len
    removed += int((~keep[1:]).sum())

    # short strokes need real Canny edges next to them (not just a frost ripple)
    support_zone = cv2.dilate(canny.astype(np.uint8) * 255, _ellipse(max(2, 0.002 * m))) > 0
    on_edge = np.bincount(labels[lines & support_zone].ravel(), minlength=n).astype(float)
    frac = np.divide(on_edge, np.maximum(length, 1))
    weak = keep & (length < 3 * min_len) & (frac < 0.35)
    weak[0] = False
    removed += int(weak.sum())
    keep &= ~weak
    lines = keep[labels] & lines

    # short strokes far from the main design are noise
    n, labels, length = _component_stats(lines)
    anchors = np.zeros(n, bool)
    anchors[1:] = length[1:] >= anchor_len
    if anchors.any():
        far = cv2.distanceTransform((~anchors[labels]).astype(np.uint8), cv2.DIST_L2, 3)
        mins = np.full(n, np.inf)
        ys, xs = np.nonzero(lines)
        np.minimum.at(mins, labels[ys, xs], far[ys, xs])
        isolated = (~anchors) & (mins > params.isolation_dist * m)
        isolated[0] = False
        removed += int(isolated.sum())
        lines = (~isolated)[labels] & lines
    return lines, removed



# ------------------------------------------------------------ gap bridging
_NEIGH = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def find_endpoints(lines: np.ndarray) -> np.ndarray:
    """(row, col) of stroke ends: line pixels with exactly one 8-neighbour."""
    k = np.ones((3, 3), np.float32)
    k[1, 1] = 0
    count = cv2.filter2D(lines.astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT)
    return np.argwhere(lines & (count == 1))


def bridge_gaps(lines: np.ndarray, max_dist: float, max_angle_deg: float = 40.0) -> tuple[np.ndarray, int]:
    """Join a stroke end to the nearest OTHER stroke it points at. Returns (lines, joins)."""
    out = lines.copy().astype(np.uint8)
    ends = find_endpoints(lines)
    if len(ends) == 0 or max_dist <= 0:
        return lines, 0
    n, labels = cv2.connectedComponents(lines.astype(np.uint8), connectivity=8)
    pts = np.argwhere(lines)
    pt_labels = labels[pts[:, 0], pts[:, 1]]
    cos_limit = np.cos(np.deg2rad(max_angle_deg))
    joins = 0
    for r, c in ends:
        lab = labels[r, c]
        near_self = pts[(pt_labels == lab) & (np.abs(pts[:, 0] - r) <= 10) & (np.abs(pts[:, 1] - c) <= 10)]
        if len(near_self) < 3:
            continue
        direction = np.array([r, c], float) - near_self.mean(axis=0)  # points away from the stroke
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            continue
        direction /= norm
        other = pts[pt_labels != lab]
        if len(other) == 0:
            continue
        delta = other - np.array([r, c])
        dist = np.hypot(delta[:, 0], delta[:, 1])
        ok = (dist > 1) & (dist <= max_dist)
        if not ok.any():
            continue
        cosang = (delta[ok] @ direction) / dist[ok]
        good = cosang >= cos_limit
        if not good.any():
            continue
        score = dist[ok][good] * (2.0 - cosang[good])  # prefer near AND well aligned
        target = other[ok][good][int(score.argmin())]
        cv2.line(out, (int(c), int(r)), (int(target[1]), int(target[0])), 1, 1)
        joins += 1
    return out.astype(bool), joins


def prune_spurs(lines: np.ndarray, min_len: int, passes: int = 2) -> tuple[np.ndarray, int]:
    """Remove short dead-end branches (spurs) hanging off longer strokes. Isolated strokes
    are not touched here (clean_lines judges them). Returns (lines, spurs_removed)."""
    out = lines.copy()
    h, w = out.shape
    k = np.ones((3, 3), np.float32)
    k[1, 1] = 0
    removed = 0
    for _ in range(passes):
        count = cv2.filter2D(out.astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT)
        ends = np.argwhere(out & (count == 1))
        changed = False
        for y, x in ends:
            path = [(y, x)]
            seen = {(y, x)}
            cy, cx = y, x
            hit_junction = False
            while len(path) <= min_len:
                nxt = [(cy + dy, cx + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                       if (dy or dx) and 0 <= cy + dy < h and 0 <= cx + dx < w
                       and out[cy + dy, cx + dx] and (cy + dy, cx + dx) not in seen]
                if not nxt:
                    break  # isolated stroke ended: leave it
                if len(nxt) > 1 or count[nxt[0]] >= 3:
                    hit_junction = True
                    break
                cy, cx = nxt[0]
                seen.add((cy, cx))
                path.append((cy, cx))
            if hit_junction and len(path) <= min_len:
                for py, px in path:
                    out[py, px] = False
                removed += 1
                changed = True
        if not changed:
            break
    return out, removed


def keep_main_body(lines: np.ndarray, reach_px: float, min_share: float = 0.08) -> tuple[np.ndarray, int]:
    """Keep strokes that belong to the big connected design; drop distant stray clusters
    (frost speckle outside the ornament). Clusters are groups of strokes within reach_px."""
    u8 = lines.astype(np.uint8)
    near = cv2.dilate(u8, _ellipse(max(1.0, reach_px / 2)))
    n, lab = cv2.connectedComponents(near, connectivity=8)
    if n <= 2:
        return lines, 0
    sizes = np.bincount(lab[lines], minlength=n)
    sizes[0] = 0
    keep = sizes >= min_share * sizes.max()
    kept = lines & keep[lab]
    return kept, int(lines.sum() - kept.sum() > 0) * (n - 1 - int(keep.sum()))


# -------------------------------------------------------------- silhouette
def _fill_holes(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, contours, -1, 255, cv2.FILLED)
    return out


def build_silhouette(lines: np.ndarray, params: OutlineParams) -> np.ndarray:
    """Region covered by the design: strokes -> bridge gaps -> fill -> shrink back."""
    m = max(lines.shape)
    grow = max(2.0, (params.gap_close * 0.5 + 0.0015) * m)
    thick = cv2.dilate(lines.astype(np.uint8) * 255, _ellipse(grow))
    closed = cv2.morphologyEx(thick, cv2.MORPH_CLOSE, _ellipse(grow))
    filled = _fill_holes(closed)
    sil = cv2.erode(filled, _ellipse(grow * 1.0))
    sil = cv2.GaussianBlur(sil, (0, 0), sigmaX=max(2.0, 0.006 * m))
    sil = ((sil > 127) * 255).astype(np.uint8)
    # keep the main body and any sizeable part, drop crumbs
    n, labels, stats, _ = cv2.connectedComponentsWithStats(sil, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = np.zeros(n, bool)
        keep[1:] = areas >= 0.05 * areas.max()
        sil = (keep[labels] * 255).astype(np.uint8)
    return sil


# --------------------------------------------------------------- rendering
def thickness_px(params: OutlineParams, m: int) -> int:
    return max(2, int(round(THICKNESS[params.line_style] * m)))


def draw_thick(thin: np.ndarray, t: int) -> np.ndarray:
    """Thicken 1-px lines to ~t px and round the edges. Returns 0/255 ink mask."""
    ink = cv2.dilate(thin.astype(np.uint8) * 255, _ellipse(max(0, (t - 1) / 2)))
    ink = cv2.GaussianBlur(ink, (0, 0), sigmaX=max(0.6, t / 4))
    return ((ink > 90) * 255).astype(np.uint8)


def draw_boundary(sil: np.ndarray, t: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Trace the silhouette's outer contours. Returns (ink, thin_line_mask, count)."""
    contours, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    thin = np.zeros_like(sil)
    cv2.drawContours(thin, contours, -1, 255, 1)
    ink = np.zeros_like(sil)
    cv2.drawContours(ink, contours, -1, 255, t, lineType=cv2.LINE_AA)
    return ((ink > 90) * 255).astype(np.uint8), thin, len(contours)


@dataclass
class OutlineOutput:
    success: bool
    outer: np.ndarray  # 0/255 ink masks, working size
    internal: np.ndarray
    combined: np.ndarray
    centerlines: np.ndarray  # 1-px strokes (debug / vectorisation)
    silhouette: np.ndarray
    crop_rgb: np.ndarray
    crop_box: tuple[int, int, int, int]
    components: int
    total_length: int
    removed: int
    outer_contours: int
    warnings: list[str] = field(default_factory=list)


def extract_outline(rgb: np.ndarray, mask: np.ndarray, params: OutlineParams | None = None) -> OutlineOutput:
    params = params or OutlineParams()
    crop, cmask, box, _scale = prepare_crop(rgb, mask, params)
    m = max(crop.shape[:2])
    enhanced, smooth = enhance(crop, params)

    sigmas = [max(1.0, s * m) for s in params.ridge_scales]
    resp, _angle = ridge_response(enhanced, sigmas, bright_only=params.bright_lines_only)  # polarity-agnostic stroke evidence

    # Etched grooves show as a bright band beside a dark band. Blurring the evidence
    # merges each pair into ONE hump, whose ridge is the stroke's centre line.
    evidence = cv2.GaussianBlur(resp, (0, 0), max(1.0, params.merge_sigma * m))
    r2, a2 = ridge_response(evidence, [max(1.0, params.merge_sigma * m)])
    nms = non_max_suppress(r2, a2)

    region = cmask > 0
    peak = float(np.percentile(evidence[region], 99))  # strength of the strongest strokes
    high = HIGH_FRAC[params.sensitivity] * peak
    low = params.low_ratio * high

    inside = cv2.erode(cmask, _ellipse(max(1, 0.002 * m))) > 0
    gate = contrast_gate(enhanced, CONTRAST_C[params.sensitivity])
    lines = hysteresis(nms & inside & gate, evidence, low, high)

    med_gray = float(np.median(smooth[region]))
    canny = cv2.Canny(smooth, int(max(10, 0.66 * med_gray * 0.5)), int(max(30, 1.33 * med_gray * 0.5))) > 0
    lines, removed = clean_lines(lines, canny, params)
    lines, joins = bridge_gaps(lines, params.bridge_dist * m)
    if joins:
        logger.info("Bridged %d gaps between stroke ends", joins)

    lines, spurs = prune_spurs(lines, max(4, int(params.spur_len * m)))
    lines, _tiny = clean_lines(lines, canny, params)
    lines, _stray = keep_main_body(lines, 0.02 * m)
    n, _, length = _component_stats(lines)
    comps, total = n - 1, int(length[1:].sum()) if n > 1 else 0
    warnings: list[str] = []
    success = total >= params.min_total_len * m
    if not success:
        warnings.append(
            f"Only {total}px of reliable line structure found (need {int(params.min_total_len * m)}px). "
            "The photo may have too little contrast, or the isolated region holds no design."
        )

    t = thickness_px(params, m)
    if total == 0:
        z = np.zeros(lines.shape, np.uint8)
        return OutlineOutput(False, z, z, z, z, z, crop, box, 0, 0, removed, 0, warnings)

    sil = build_silhouette(lines, params)
    outer, outer_thin, n_contours = draw_boundary(sil, t)

    # interior = strokes not hugging the boundary (the boundary line already shows those)
    band = max(t * 1.5, params.boundary_band * m)
    near_border = cv2.dilate(outer_thin, _ellipse(band)) > 0
    interior = lines & ~near_border
    # drop interior fragments that the border removal left too short to matter
    n2, labels2, length2 = _component_stats(interior)
    keep2 = np.zeros(n2, bool)
    keep2[1:] = length2[1:] >= max(4, int(0.6 * params.min_component_len * m))
    interior = keep2[labels2] & interior
    internal = draw_thick(interior, t)
    combined = cv2.bitwise_or(outer, internal)
    return OutlineOutput(success, outer, internal, combined, (lines * 255).astype(np.uint8), sil,
                         crop, box, comps, total, removed, n_contours, warnings)
