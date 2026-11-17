"""Segmentation tests on SYNTHETIC scenes with known ground truth.

These show the method behaves as designed on controlled images. They do not prove
accuracy on real photographs, and SAM 2 is only exercised through a stand-in
predictor (the real model is not installed in the test environment)."""
import json

import numpy as np
import pytest
from PIL import Image

from agents.segmentation_agent import SegmentationAgent
from config import settings
from schemas.analysis import ImageAnalysis
from schemas.segmentation import SegmentationParams
from services.segmentation_service import (
    Sam2Segmenter,
    Sam2UnavailableError,
    classical_segment,
    expand_roi,
    mask_bbox_px,
    mask_iou,
)
from synth import etched_scene

BAD_BOX = (0.35, 0.0, 0.55, 0.85)  # shifted left of the ornament, like a real Groq answer


def coverage(mask, truth):
    return float(((mask > 0) & (truth > 0)).sum()) / float((truth > 0).sum())


def leak(mask, other):
    return float(((mask > 0) & (other > 0)).sum()) / max(float((mask > 0).sum()), 1.0)


@pytest.fixture(scope="module")
def scene():
    return etched_scene()


def analysis(box=BAD_BOX, sensitivity="medium"):
    return ImageAnalysis.model_validate(
        {"primary_object": "etched ornament", "confidence": 0.9, "bounding_box": box,
         "extraction_hints": {"edge_sensitivity": sensitivity}}
    )


@pytest.fixture()
def photo(tmp_path, scene):
    p = tmp_path / "scene.png"
    Image.fromarray(scene[0]).save(p)
    return p


@pytest.fixture(autouse=True)
def out_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "output_dir", tmp_path / "out")
    monkeypatch.setattr(settings, "segmentation_backend", "classical")


# ------------------------------------------------------------ classical core
@pytest.mark.parametrize("box", [BAD_BOX, (0.48, 0.0, 0.76, 0.9), None])
def test_finds_ornament_and_ignores_person(scene, box):
    rgb, ornament, person = scene
    out = classical_segment(rgb, box, SegmentationParams())
    assert out.success
    assert coverage(out.mask, ornament) > 0.9
    assert leak(out.mask, person) < 0.02


def test_mask_has_original_size_after_downscaling():
    rgb, ornament, _ = etched_scene(h=2400, w=1800)  # larger than working_max_side
    out = classical_segment(rgb, BAD_BOX, SegmentationParams())
    assert out.mask.shape == rgb.shape[:2] and out.line_map.shape == rgb.shape[:2]
    assert out.success and coverage(out.mask, ornament) > 0.9


def test_chooses_region_that_holds_the_boxed_lines():
    """With two ornaments, the box decides which one is selected."""
    rgb, orn_a, _ = etched_scene(with_person=False)
    rgb_b, orn_b, _ = etched_scene(with_person=False)
    shifted = np.roll(rgb_b, -400, axis=1)  # second copy far to the left
    both = np.minimum(rgb, shifted)
    out = classical_segment(both, (0.50, 0.0, 0.74, 0.9), SegmentationParams())
    assert out.success
    ys, xs = np.where(out.mask > 0)
    assert xs.mean() / both.shape[1] > 0.5  # picked the right-hand ornament


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_plain_frost_is_reported_as_failure(seed):
    rgb, _, _ = etched_scene(seed=seed, with_person=False, with_ornament=False)
    out = classical_segment(rgb, (0.3, 0.1, 0.6, 0.8), SegmentationParams())
    assert not out.success and out.warnings and out.confidence == 0.0


def test_flat_image_is_reported_as_failure():
    out = classical_segment(np.full((400, 600, 3), 200, np.uint8), None, SegmentationParams())
    assert not out.success and "No line structure" in out.warnings[0]


def test_failure_mask_is_search_area_not_whole_image():
    rgb, _, _ = etched_scene(with_person=False, with_ornament=False)
    out = classical_segment(rgb, (0.3, 0.1, 0.6, 0.8), SegmentationParams())
    assert not out.success
    assert 0.05 < (out.mask > 0).mean() < 0.9


def test_expand_roi_clamps_and_keeps_minimum_margin():
    p = SegmentationParams()
    x1, y1, x2, y2 = expand_roi((0.4, 0.4, 0.45, 0.45), p)
    assert (x2 - x1) >= 0.05 + 2 * p.roi_min_margin - 1e-9
    assert expand_roi((0.0, 0.0, 1.0, 1.0), p) == (0.0, 0.0, 1.0, 1.0)
    assert expand_roi(None, p) is None


def test_sensitivity_levels_all_work(scene):
    rgb, ornament, _ = scene
    for s in ("low", "medium", "high"):
        out = classical_segment(rgb, BAD_BOX, SegmentationParams(sensitivity=s))
        assert out.success and coverage(out.mask, ornament) > 0.8, s


# ---------------------------------------------------------------------- agent
def test_agent_writes_all_outputs(photo, scene):
    result = SegmentationAgent().run("job1", photo, analysis())
    d = settings.output_dir / "job1"
    for name in ("mask.png", "isolated.png", "isolated_crop.png", "line_map.png", "debug_overlay.png", "segmentation.json"):
        assert (d / name).is_file(), name
    assert result.success and result.method == "classical" and result.sam2_status == "not_requested"
    saved = json.loads((d / "segmentation.json").read_text())
    assert saved["success"] is True and saved["image_size"] == [750, 1000]
    assert Image.open(d / "mask.png").size == (750, 1000)


def test_isolated_image_is_white_outside_mask(photo):
    SegmentationAgent().run("job2", photo, analysis())
    d = settings.output_dir / "job2"
    mask = np.asarray(Image.open(d / "mask.png"))
    iso = np.asarray(Image.open(d / "isolated.png"))
    assert (iso[mask == 0] == 255).all()
    assert (iso[mask > 0] != 255).any()


def test_agent_reports_failure_honestly(tmp_path):
    rgb, _, _ = etched_scene(with_person=False, with_ornament=False)
    p = tmp_path / "plain.png"
    Image.fromarray(rgb).save(p)
    result = SegmentationAgent().run("bad", p, analysis((0.3, 0.1, 0.6, 0.8)))
    assert not result.success and result.method == "roi_box_fallback" and result.confidence == 0.0
    assert any("failed" in w.lower() for w in result.warnings)
    assert (settings.output_dir / "bad" / "debug_overlay.png").is_file()


def test_agent_works_without_ai_box(photo):
    result = SegmentationAgent().run("nobox", photo, analysis(box=None))
    assert result.success and result.roi_bbox is None
    assert result.mask_bbox[0] > 0.4  # ornament is right of centre


def test_agent_uses_exif_orientation(tmp_path, scene):
    rgb = scene[0]
    rotated = np.rot90(rgb, k=1)  # stored sideways ...
    img = Image.fromarray(np.ascontiguousarray(rotated))
    exif = Image.Exif()
    exif[0x0112] = 8  # ... with a flag that rotates it back for display
    p = tmp_path / "sideways.jpg"
    img.save(p, "JPEG", exif=exif, quality=95)
    result = SegmentationAgent().run("exif", p, analysis())
    assert result.image_size == (750, 1000)  # width x height after correction


def test_retry_params_change_the_result(photo):
    strict = SegmentationParams(density_scale=2.0, grow_reach=0.0)
    loose = SegmentationParams(density_scale=1.0, grow_reach=0.2)
    a = SegmentationAgent().run("strict", photo, analysis(), strict)
    b = SegmentationAgent().run("loose", photo, analysis(), loose)
    assert b.area_fraction > a.area_fraction


# ----------------------------------------------------------------------- SAM 2
class FakePredictor:
    """Stands in for SAM2ImagePredictor (set_image / predict with a box prompt)."""

    def __init__(self, mask_fn):
        self.mask_fn = mask_fn
        self.calls = []

    def set_image(self, rgb):
        self.shape = rgb.shape[:2]

    def predict(self, box, multimask_output):
        self.calls.append((np.asarray(box), multimask_output))
        good = self.mask_fn(self.shape, np.asarray(box)[0])
        junk = np.zeros(self.shape, np.uint8)
        return np.stack([junk, good, junk]), np.array([0.1, 0.9, 0.2]), None


def classical_like(scene_rgb):
    def fn(shape, box):
        return classical_segment(scene_rgb, BAD_BOX, SegmentationParams()).mask
    return fn


def test_sam2_unavailable_falls_back_and_says_so(photo, monkeypatch):
    monkeypatch.setattr(settings, "segmentation_backend", "auto")
    sam = Sam2Segmenter(checkpoint="", model_cfg="x")
    result = SegmentationAgent(sam2=sam).run("auto", photo, analysis())
    assert result.success and result.method == "classical"
    assert result.sam2_status.startswith("unavailable") and "SAM2_CHECKPOINT" in result.sam2_status


def test_sam2_explicitly_requested_but_missing_adds_warning(photo, monkeypatch):
    monkeypatch.setattr(settings, "segmentation_backend", "sam2")
    result = SegmentationAgent(sam2=Sam2Segmenter("", "x")).run("req", photo, analysis())
    assert result.method == "classical" and any("SAM 2 was requested" in w for w in result.warnings)


def test_sam2_missing_package_is_reported(tmp_path):
    ckpt = tmp_path / "w.pt"
    ckpt.write_bytes(b"x")
    ok, why = Sam2Segmenter(str(ckpt), "cfg").availability()
    assert not ok and "not installed" in why
    with pytest.raises(Sam2UnavailableError):
        Sam2Segmenter(str(ckpt), "cfg").segment(np.zeros((10, 10, 3), np.uint8), (0, 0, 5, 5))


def test_sam2_used_when_it_agrees(photo, scene, monkeypatch):
    monkeypatch.setattr(settings, "segmentation_backend", "auto")
    fake = FakePredictor(classical_like(scene[0]))
    result = SegmentationAgent(sam2=Sam2Segmenter("", "x", predictor=fake)).run("sam_ok", photo, analysis())
    assert result.method == "sam2" and result.sam2_status == "used"
    box, multimask = fake.calls[0]
    assert box.shape == (1, 4) and multimask is True  # documented box-prompt call
    assert (box[0][2] > box[0][0]) and (box[0][3] > box[0][1])


def test_sam2_rejected_when_it_grabs_the_whole_pane(photo, monkeypatch):
    monkeypatch.setattr(settings, "segmentation_backend", "auto")
    fake = FakePredictor(lambda shape, box: np.full(shape, 255, np.uint8))
    result = SegmentationAgent(sam2=Sam2Segmenter("", "x", predictor=fake)).run("sam_bad", photo, analysis())
    assert result.method == "classical" and result.sam2_status.startswith("rejected")


def test_sam2_not_run_when_classical_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "segmentation_backend", "auto")
    rgb, _, _ = etched_scene(with_person=False, with_ornament=False)
    p = tmp_path / "plain.png"
    Image.fromarray(rgb).save(p)
    fake = FakePredictor(lambda shape, box: np.full(shape, 255, np.uint8))
    result = SegmentationAgent(sam2=Sam2Segmenter("", "x", predictor=fake)).run("sam_skip", p, analysis((0.3, 0.1, 0.6, 0.8)))
    assert not result.success and result.sam2_status.startswith("not_run") and fake.calls == []


def test_mask_helpers():
    a = np.zeros((10, 10), np.uint8); a[2:6, 2:6] = 255
    b = np.zeros((10, 10), np.uint8); b[4:8, 4:8] = 255
    assert mask_iou(a, a) == 1.0 and 0 < mask_iou(a, b) < 1 and mask_iou(a, np.zeros_like(a)) == 0.0
    assert mask_bbox_px(a) == (2, 2, 6, 6) and mask_bbox_px(np.zeros_like(a)) is None
