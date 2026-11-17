"""Outline tests on SYNTHETIC scenes with known strokes. They show the method works on
controlled images; they do not prove quality on real photographs."""
import json

import cv2
import numpy as np
import pytest
from PIL import Image

from agents.outline_extraction_agent import OutlineExtractionAgent
from config import settings
from schemas.analysis import ImageAnalysis
from schemas.outline import OutlineParams
from services.outline_service import extract_outline, prepare_crop
from services.segmentation_service import classical_segment
from schemas.segmentation import SegmentationParams
from synth import etched_scene

BOX = (0.35, 0.0, 0.55, 0.85)


@pytest.fixture(scope="module")
def scene():
    rgb, _, _ = etched_scene()
    base, _, _ = etched_scene(with_ornament=False)
    truth = (np.abs(rgb.astype(int) - base.astype(int)).sum(2) > 60)
    mask = classical_segment(rgb, BOX, SegmentationParams()).mask
    return rgb, mask, truth


@pytest.fixture(autouse=True)
def out_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "output_dir", tmp_path / "out")


def test_strokes_found_with_high_recall_and_precision(scene):
    rgb, mask, truth = scene
    out = extract_outline(rgb, mask, OutlineParams())
    assert out.success
    crop, _, box, scale = prepare_crop(rgb, mask, OutlineParams())
    x1, y1, x2, y2 = box
    gt = cv2.resize(truth[y1:y2, x1:x2].astype(np.uint8), (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    tol = int(6 * scale)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    lines = out.centerlines > 0
    precision = (lines & (cv2.dilate(gt.astype(np.uint8), k) > 0)).sum() / lines.sum()
    recall = (gt & (cv2.dilate(lines.astype(np.uint8), k) > 0)).sum() / gt.sum()
    assert precision > 0.9 and recall > 0.9


def test_plain_frost_reports_failure():
    rgb, _, _ = etched_scene(with_person=False, with_ornament=False)
    mask = np.zeros(rgb.shape[:2], np.uint8)
    mask[100:900, 200:550] = 255
    out = extract_outline(rgb, mask, OutlineParams())
    assert not out.success and out.warnings


def test_outputs_are_binary_and_keep_aspect(scene):
    rgb, mask, _ = scene
    out = extract_outline(rgb, mask, OutlineParams())
    assert set(np.unique(out.combined)) <= {0, 255}
    x1, y1, x2, y2 = out.crop_box
    h, w = out.combined.shape
    assert abs((w / h) - ((x2 - x1) / (y2 - y1))) < 0.02


def test_line_style_changes_thickness(scene):
    rgb, mask, _ = scene
    thin = extract_outline(rgb, mask, OutlineParams(line_style="thin")).combined
    bold = extract_outline(rgb, mask, OutlineParams(line_style="bold")).combined
    assert (bold > 0).sum() > (thin > 0).sum()


def test_agent_writes_files_black_on_white(tmp_path, scene):
    rgb, mask, _ = scene
    p = tmp_path / "s.png"
    Image.fromarray(rgb).save(p)
    analysis = ImageAnalysis.model_validate({"primary_object": "x", "confidence": 0.9, "bounding_box": BOX})
    res = OutlineExtractionAgent().run("j1", p, mask, analysis)
    d = settings.output_dir / "j1"
    for n in ("outer_boundary.png", "internal_design.png", "combined_outline.png", "centerlines.png", "outline_preview.png", "outline.json"):
        assert (d / n).is_file(), n
    img = np.asarray(Image.open(d / "combined_outline.png"))
    assert set(np.unique(img)) <= {0, 255} and (img == 0).any() and (img == 255).mean() > 0.5
    assert json.loads((d / "outline.json").read_text())["success"] is res.success is True


def test_agent_with_empty_mask_fails_honestly(tmp_path, scene):
    rgb, _, _ = scene
    p = tmp_path / "s.png"
    Image.fromarray(rgb).save(p)
    res = OutlineExtractionAgent().run("j2", p, np.zeros(rgb.shape[:2], np.uint8))
    assert not res.success and res.warnings


def test_bright_only_is_default_and_finds_bright_strokes(scene):
    assert OutlineParams().bright_lines_only is True
    rgb, mask, _ = scene
    both = extract_outline(rgb, mask, OutlineParams(bright_lines_only=False)).centerlines
    assert extract_outline(rgb, mask, OutlineParams()).success and (both > 0).any()


def test_spur_pruning_removes_short_branches_only():
    from services.outline_service import prune_spurs
    a = np.zeros((100, 100), bool)
    a[50, 10:90] = True  # long stroke
    a[44:50, 50] = True  # 6px spur
    out, n = prune_spurs(a, 10)
    assert n >= 1 and not out[44:49, 50].any() and out[50, 10:90].all()
